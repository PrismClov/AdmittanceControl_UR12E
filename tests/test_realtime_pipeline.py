import socket
import threading
import time
import unittest

from realtime_pipeline import LatestRobotState, ServoPStream


class FakeFeedback:
    def __init__(self):
        self.value = 0

    def feedBackData(self):
        time.sleep(0.005)
        self.value += 1
        return {"value": self.value}


class LatestRobotStateTest(unittest.TestCase):
    def test_snapshot_returns_latest_without_waiting_for_sequence(self):
        reader = LatestRobotState(FakeFeedback())
        reader.start()
        try:
            first, sequence, _ = reader.wait_for_data()
            time.sleep(0.02)
            latest, latest_sequence, _ = reader.snapshot()
            self.assertGreaterEqual(latest_sequence, sequence)
            self.assertGreaterEqual(latest["value"], first["value"])
            stats = reader.period_stats()
            self.assertGreater(stats["samples"], 0)
            self.assertGreater(stats["median_hz"], 50.0)
        finally:
            reader.stop()


class ServoPStreamTest(unittest.TestCase):
    def test_sender_and_reply_reader_are_independent(self):
        client, server = socket.socketpair()
        received = []
        stop_server = threading.Event()

        def serve():
            buffer = ""
            server.settimeout(0.05)
            while not stop_server.is_set():
                try:
                    chunk = server.recv(4096)
                except TimeoutError:
                    continue
                if not chunk:
                    return
                buffer += chunk.decode()
                messages = buffer.split(")")
                buffer = messages.pop()
                for message in messages:
                    if "ServoP(" in message:
                        received.append(message + ")")
                        server.sendall(b"0,{1},ServoP();")

        server_thread = threading.Thread(target=serve, daemon=True)
        server_thread.start()
        stream = ServoPStream(client, period_s=0.01)
        stream.start()
        try:
            for _ in range(20):
                stream.update_target([1, 2, 3, 4, 5, 6])
                time.sleep(0.005)
            stream.raise_if_fault()
            self.assertGreaterEqual(stream.sent_count, 3)
            self.assertGreaterEqual(len(stream.reply_latencies), 3)
            self.assertEqual(stream.unknown_reply_count, 0)
        finally:
            stream.stop()
            stop_server.set()
            server_thread.join(timeout=1.0)
            client.close()
            server.close()


if __name__ == "__main__":
    unittest.main()
