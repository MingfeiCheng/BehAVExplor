#
# Copyright (c) 2019-2020 LG Electronics, Inc.
#
# This software contains code licensed as described in LICENSE.
#

import threading
import websockets
import asyncio
import json

from multiprocessing import Queue
from loguru import logger

class Remote(threading.Thread):

    def __init__(self, host, port):
        super().__init__(daemon=True)
        self.endpoint = "ws://{}:{}".format(host, port)
        self.lock = threading.Lock()
        self.cv = threading.Condition()
        self.data = None
        self.sem = threading.Semaphore(0)
        self.running = True
        self.start()
        self.sem.acquire()

        self.sim_data = None
        self.state_queue = Queue()

    def run(self):
        logger.info('Remote: run')
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self.process())

    def close(self):
        asyncio.run_coroutine_threadsafe(self.websocket.close(), self.loop)
        self.join()
        self.loop.close()

    async def process(self):
        self.websocket = await websockets.connect(self.endpoint, compression=None)
        self.sem.release()

        while True:
            try:
                data = await self.websocket.recv() # continuous listen
                # logger.info(data)
            except Exception as e:
                if isinstance(e, websockets.exceptions.ConnectionClosed):
                    break
                with self.cv:
                    self.data = {"error": str(e)}
                    self.cv.notify()
                break

            with self.cv:
                self.data = json.loads(data) # if running -> None, will overwrite last one
                if self.data is not None:
                    # logger.warning(self.data)
                    if ('result' in self.data.keys()) and (self.data['result'] is not None) and isinstance(self.data['result'], dict) and "events" in self.data['result']:
                        # logger.error(self.data)
                        self.sim_data = self.data
                        # self.data = None
                self.cv.notify()

        logger.error('Close websocket')
        await self.websocket.close()

    def command(self, name, args={}):
        if not self.websocket:
            raise Exception("Not connected")

        input_data = json.dumps({"command": name, "arguments": args})

        if self.data is not None:
            self.sim_data = self.data
            self.data = None

        asyncio.run_coroutine_threadsafe(self.websocket.send(input_data), self.loop)

        with self.cv:
            if name == 'agent/state/get':
                self.cv.wait_for(lambda: (self.data is not None and isinstance(self.data, dict)) and ('result' in self.data.keys()) and (self.data['result'] is not None) and ("events" not in self.data['result']))
                if isinstance(self.data['result'], dict) and "event" in self.data['result']:
                    logger.error(self.data)
            else:
                self.cv.wait_for(lambda: self.data is not None)
            output_data = self.data
            self.data = None

        if "error" in output_data:
            raise Exception(output_data["error"])
        return output_data["result"]

    def status_monitor(self):
        if not self.websocket:
            raise Exception("Not connected")

        if self.sim_data is not None:
            if "error" in self.sim_data:
                raise Exception(self.sim_data["error"])
            return self.sim_data["result"]
        else:
            self.sim_data = self.data
            self.data = None
            if self.sim_data is None:
                return self.sim_data
            else:
                if "error" in self.sim_data:
                    raise Exception(self.sim_data["error"])
                return self.sim_data["result"]

    def command_run(self, name, args={}):
        if not self.websocket:
            raise Exception("Not connected")
        logger.info('[PythonAPI] Start Running')
        data = json.dumps({"command": name, "arguments": args})
        asyncio.run_coroutine_threadsafe(self.websocket.send(data), self.loop)
        self.sim_data = None
        self.data = None
        return None