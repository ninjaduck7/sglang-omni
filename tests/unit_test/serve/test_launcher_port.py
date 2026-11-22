import socket

from sglang_omni.serve.launcher import _find_available_port


def test_port_after_active_close_and_live_listener():
    host = "127.0.0.1"
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((host, 0))
        listener.listen()
        port = listener.getsockname()[1]
        assert _find_available_port(host, port) != port
        with socket.create_connection((host, port)) as client:
            accepted, _ = listener.accept()
            accepted.close()
            assert client.recv(1) == b""
    assert _find_available_port(host, port) == port
