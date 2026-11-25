import json
import os
from pathlib import Path
import re
import shutil
import socket
import sys
import tempfile
import threading
import time

import pytest

from dipy.desktop.constants import MAX_FILES
from dipy.desktop.ipc import (
    _CLIENT_LABEL,
    _PROTOCOL,
    _REPLY_LABEL,
    _SERVER_LABEL,
    WindowServer,
    _Server,
    _bind,
    _claim,
    _connect,
    _current_user_sid,
    _endpoint,
    _exchange,
    _nonce,
    _proof,
    _recv_frame,
    _request,
    _send_frame,
    collect_batch,
    ipc_authkey,
    live_windows,
    send_open,
)


@pytest.fixture
def ipc_dir():
    # The temporary folders of pytest can exceed the length of a socket path.
    base = Path(tempfile.mkdtemp(prefix="sk", dir=None if os.name == "nt" else "/tmp"))
    yield base / "ipc"
    shutil.rmtree(base, ignore_errors=True)


def _touch(path):
    path.write_bytes(b"")
    return path


def test_window_receives_checked_paths_in_order(ipc_dir, tmp_path):
    volume = _touch(tmp_path / "fa.nii.gz")
    tractogram = _touch(tmp_path / "bundle.trk")
    server = WindowServer.start(ipc_dir=ipc_dir)
    try:
        (window,) = live_windows(ipc_dir=ipc_dir)
        assert window.number == 1

        assert send_open(window, [volume, tractogram], ipc_dir=ipc_dir)
        assert server.receive() == [str(volume.absolute()), str(tractogram.absolute())]
        assert server.receive() == []
    finally:
        server.close()


def test_windows_report_which_one_was_used_last(ipc_dir, tmp_path):
    volume = _touch(tmp_path / "fa.nii.gz")
    first = WindowServer.start(ipc_dir=ipc_dir)
    second = WindowServer.start(ipc_dir=ipc_dir)
    try:
        time.sleep(0.05)
        first.touch()
        used_last = max(live_windows(ipc_dir=ipc_dir), key=lambda w: w.last_active)
        assert used_last.number == first.number

        time.sleep(0.05)
        window_2 = live_windows(ipc_dir=ipc_dir)[1]
        assert send_open(window_2, [volume], ipc_dir=ipc_dir)
        used_last = max(live_windows(ipc_dir=ipc_dir), key=lambda w: w.last_active)
        assert used_last.number == second.number
    finally:
        first.close()
        second.close()


def test_closed_window_disappears_and_its_number_is_reused(ipc_dir, tmp_path):
    volume = _touch(tmp_path / "fa.nii.gz")
    first = WindowServer.start(ipc_dir=ipc_dir)
    second = WindowServer.start(ipc_dir=ipc_dir)
    third = None
    try:
        assert (first.number, second.number) == (1, 2)
        stale = live_windows(ipc_dir=ipc_dir)[0]

        first.close()
        assert [w.number for w in live_windows(ipc_dir=ipc_dir)] == [2]

        third = WindowServer.start(ipc_dir=ipc_dir)
        assert third.number == 1
        assert not send_open(stale, [volume], ipc_dir=ipc_dir)
        assert third.receive() == []
    finally:
        for server in (first, second, third):
            if server is not None:
                server.close()


def test_window_refuses_files_it_cannot_open(ipc_dir, tmp_path):
    notes = tmp_path / "notes.txt"
    notes.write_text("not an image", encoding="utf-8")
    server = WindowServer.start(ipc_dir=ipc_dir)
    try:
        (window,) = live_windows(ipc_dir=ipc_dir)

        assert not send_open(window, [tmp_path / "missing.nii.gz"], ipc_dir=ipc_dir)
        assert not send_open(window, [notes], ipc_dir=ipc_dir)
        assert server.receive() == []
    finally:
        server.close()


def test_window_takes_nothing_from_a_launch_with_the_wrong_key(ipc_dir, tmp_path):
    volume = _touch(tmp_path / "fa.nii.gz")
    server = WindowServer.start(ipc_dir=ipc_dir)
    key_file = ipc_dir / "authkey"
    original = key_file.read_bytes()
    try:
        (window,) = live_windows(ipc_dir=ipc_dir)

        key_file.write_bytes(b"\x01" * 32)
        assert key_file.read_bytes() != original
        assert not send_open(window, [volume], ipc_dir=ipc_dir)
        assert server.receive() == []

        key_file.write_bytes(original)
        assert send_open(window, [volume], ipc_dir=ipc_dir)
        assert server.receive() == [str(volume.absolute())]
    finally:
        server.close()


def test_silent_connection_blocks_neither_requests_nor_close(ipc_dir, tmp_path):
    volume = _touch(tmp_path / "fa.nii.gz")
    server = WindowServer.start(ipc_dir=ipc_dir)
    silent = _connect("window-1", ipc_dir)
    try:
        (window,) = live_windows(ipc_dir=ipc_dir)
        assert send_open(window, [volume], ipc_dir=ipc_dir)
    finally:
        started = time.monotonic()
        server.close()
        silent.close()
    assert time.monotonic() - started < 3


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permissions")
def test_key_and_folder_are_private(ipc_dir):
    ipc_authkey(ipc_dir=ipc_dir)

    assert ipc_dir.stat().st_mode & 0o777 == 0o700
    assert (ipc_dir / "authkey").stat().st_mode & 0o777 == 0o600


def test_simultaneous_launches_merge_into_one(ipc_dir, tmp_path):
    first = str(_touch(tmp_path / "a.nii.gz"))
    second = str(_touch(tmp_path / "b.trk"))
    results = {}

    def launch(name, path):
        results[name] = collect_batch([path], ipc_dir=ipc_dir, quiet=1.0)

    threads = [
        threading.Thread(target=launch, args=("first", first)),
        threading.Thread(target=launch, args=("second", second)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(15)

    merged = [paths for paths in results.values() if paths is not None]
    assert len(results) == 2 and len(merged) == 1
    assert set(merged[0]) == {first, second}


def test_a_lone_launch_keeps_its_files(ipc_dir, tmp_path):
    path = str(_touch(tmp_path / "a.nii.gz"))

    assert collect_batch([path], ipc_dir=ipc_dir, quiet=0.1) == [path]


def test_batch_refuses_files_beyond_the_limit(ipc_dir, tmp_path):
    first = str(_touch(tmp_path / "a.nii.gz"))
    extra = [str(_touch(tmp_path / f"{i}.nii.gz")) for i in range(MAX_FILES)]
    results = []
    collector = threading.Thread(
        target=lambda: results.append(
            collect_batch([first], ipc_dir=ipc_dir, quiet=1.0)
        )
    )
    collector.start()
    deadline = time.monotonic() + 5
    while not _endpoint("launch", ipc_dir).exists() and time.monotonic() < deadline:
        time.sleep(0.01)

    reply = None
    while reply is None and time.monotonic() < deadline:
        reply = _request("launch", {"op": "collect", "paths": extra}, ipc_dir)
    collector.join(15)

    assert reply == {"ok": False}
    assert results == [[first]]


def _handshake(sock, key):
    deadline = time.monotonic() + 3
    client_nonce = _nonce()
    _send_frame(sock, {"protocol": _PROTOCOL, "nonce": client_nonce})
    hello = _recv_frame(sock, deadline, 512)
    assert hello["proof"] == _proof(key, _SERVER_LABEL, client_nonce, hello["nonce"])
    return client_nonce, hello["nonce"], deadline


def test_window_ignores_a_request_that_does_not_match_its_proof(ipc_dir, tmp_path):
    volume = _touch(tmp_path / "fa.nii.gz")
    server = WindowServer.start(ipc_dir=ipc_dir)
    key = ipc_authkey(ipc_dir=ipc_dir)
    try:
        request = {"op": "open", "instance": server.instance, "paths": [str(volume)]}
        body = json.dumps(request)

        with _connect("window-1", ipc_dir) as sock:
            client_nonce, server_nonce, deadline = _handshake(sock, key)
            other = json.dumps({"op": "describe"})
            proof = _proof(key, _CLIENT_LABEL, server_nonce, client_nonce, body=other)
            _send_frame(sock, {"body": body, "proof": proof})
            with pytest.raises(ConnectionError):
                _recv_frame(sock, deadline, 1 << 20)
        assert server.receive() == []

        with _connect("window-1", ipc_dir) as sock:
            client_nonce, server_nonce, deadline = _handshake(sock, key)
            proof = _proof(key, _CLIENT_LABEL, server_nonce, client_nonce, body=body)
            _send_frame(sock, {"body": body, "proof": proof})
            answer = _recv_frame(sock, deadline, 1 << 20)
        assert json.loads(answer["body"]) == {"ok": True}
        assert server.receive() == [str(volume.absolute())]
    finally:
        server.close()


def _fake_server(sock, key, reply, proved_reply):
    deadline = time.monotonic() + 3
    hello = _recv_frame(sock, deadline, 512)
    server_nonce = _nonce()
    proof = _proof(key, _SERVER_LABEL, hello["nonce"], server_nonce)
    _send_frame(sock, {"nonce": server_nonce, "proof": proof})
    _recv_frame(sock, deadline, 1 << 20)
    proof = _proof(key, _REPLY_LABEL, server_nonce, hello["nonce"], body=proved_reply)
    _send_frame(sock, {"body": reply, "proof": proof})


def test_launch_rejects_a_reply_that_does_not_match_its_proof():
    key = os.urandom(32)
    accepted = json.dumps({"ok": True})
    refused = json.dumps({"ok": False})
    for reply, proved_reply in ((accepted, accepted), (accepted, refused)):
        client, server = socket.socketpair()
        thread = threading.Thread(
            target=_fake_server, args=(server, key, reply, proved_reply)
        )
        thread.start()
        try:
            if reply == proved_reply:
                assert _exchange(client, key, {"op": "describe"}) == {"ok": True}
            else:
                with pytest.raises(ValueError, match="reply proof"):
                    _exchange(client, key, {"op": "describe"})
        finally:
            thread.join(5)
            client.close()
            server.close()


def _serve_description(ipc_dir, description):
    key = ipc_authkey(ipc_dir=ipc_dir)
    lock_file = _claim("window-1", ipc_dir)
    return _Server(
        name="window-1",
        lock_file=lock_file,
        sock=_bind("window-1", ipc_dir),
        key=key,
        handler=lambda request: description,
        ipc_dir=ipc_dir,
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"pid": -1},
        {"pid": True},
        {"pid": 2**40},
        {"number": True},
        {"number": 2},
        {"last_active": float("nan")},
        {"last_active": float("inf")},
        {"last_active": 10**400},
        {"last_active": "now"},
        {"instance": 7},
    ],
)
def test_live_windows_refuses_an_implausible_description(ipc_dir, changes):
    honest = {"number": 1, "pid": 4242, "instance": "abc", "last_active": 1.5}
    server = _serve_description(ipc_dir, {**honest, **changes})
    try:
        assert live_windows(ipc_dir=ipc_dir) == []
    finally:
        server.close()


def test_live_windows_accepts_a_plausible_description(ipc_dir):
    honest = {"number": 1, "pid": 4242, "instance": "abc", "last_active": 1.5}
    server = _serve_description(ipc_dir, honest)
    try:
        (window,) = live_windows(ipc_dir=ipc_dir)
        assert (window.pid, window.instance, window.last_active) == (4242, "abc", 1.5)
    finally:
        server.close()


def test_endpoint_of_a_dead_window_is_removed(ipc_dir):
    ipc_authkey(ipc_dir=ipc_dir)
    stale = _endpoint("window-5", ipc_dir)
    stale.write_text("1", encoding="utf-8")

    assert live_windows(ipc_dir=ipc_dir) == []
    assert not stale.exists()


@pytest.mark.parametrize("name", ["quiet", "limit"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0])
def test_collect_batch_refuses_unusable_waiting_times(ipc_dir, tmp_path, name, value):
    path = str(_touch(tmp_path / "a.nii.gz"))

    with pytest.raises(ValueError, match=name):
        collect_batch([path], ipc_dir=ipc_dir, **{name: value})


@pytest.mark.skipif(sys.platform != "win32", reason="Windows lets ports be shared")
def test_window_port_cannot_be_shared_by_another_socket(ipc_dir):
    server = WindowServer.start(ipc_dir=ipc_dir)
    try:
        port = int(_endpoint("window-1", ipc_dir).read_text(encoding="utf-8"))
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as other:
            other.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            with pytest.raises(OSError):
                other.bind(("127.0.0.1", port))
    finally:
        server.close()


def test_window_refuses_files_that_arrive_after_it_closed(ipc_dir, tmp_path):
    volume = _touch(tmp_path / "fa.nii.gz")
    server = WindowServer.start(ipc_dir=ipc_dir)
    request = {"op": "open", "instance": server.instance, "paths": [str(volume)]}
    server.close()

    assert server._handle(request)["ok"] is False
    assert server.receive() == []


def test_launches_go_on_when_a_lock_file_cannot_be_opened(ipc_dir, tmp_path):
    path = str(_touch(tmp_path / "a.nii.gz"))
    ipc_authkey(ipc_dir=ipc_dir)
    _endpoint("window-1", ipc_dir).write_text("1", encoding="utf-8")
    (ipc_dir / "window-1.lock").mkdir()
    (ipc_dir / "launch.lock").mkdir()

    assert live_windows(ipc_dir=ipc_dir) == []
    assert collect_batch([path], ipc_dir=ipc_dir) == [path]


@pytest.mark.skipif(sys.platform != "win32", reason="Security identifiers are Windows")
def test_current_user_sid_is_a_security_identifier():
    assert re.fullmatch(r"S-1-\d+(-\d+)+", _current_user_sid())


def test_launches_that_start_together_share_one_key(ipc_dir):
    keys = []
    launches = [
        threading.Thread(target=lambda: keys.append(ipc_authkey(ipc_dir=ipc_dir)))
        for _ in range(6)
    ]
    for launch in launches:
        launch.start()
    for launch in launches:
        launch.join(60)

    assert len(keys) == 6
    assert len(set(keys)) == 1 and keys[0] == (ipc_dir / "authkey").read_bytes()
