"""Skyline windows that accept files from other launches.

Each desktop Skyline window listens on a local endpoint inside a folder of the
user and can be asked to open more files. Requests are authenticated with a
secret key kept in that folder and a handshake that binds each proof to its
role and to the session: a client never sends a request to a server that does
not know the key, a server never acts on a request that was not sent by a
client that knows it, and the reply is vouched for the same way, so a process
sitting on the endpoint can neither alter a request nor forge a reply.

The transport is a plain ``socket`` with its own framing: ``AF_UNIX`` sockets
inside the private folder on POSIX; on Windows, where Python has no
``AF_UNIX``, TCP on ``127.0.0.1`` with the port published in a file of that
folder. ``multiprocessing.connection`` is not used: its handshake runs without
a timeout and its Windows pipes carry no access list.

This module must stay light: it imports only the standard library and the
light modules of ``dipy.desktop``.
"""

from dataclasses import dataclass
import hmac
import json
import math
import os
from pathlib import Path
import queue
import re
import secrets
import socket
import subprocess
import sys
import tempfile
import threading
import time

from dipy.desktop.constants import MAX_FILES, SKYLINE_HOME
from dipy.desktop.links import checked_file
from dipy.utils.logging import logger

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl

IPC_DIR = SKYLINE_HOME / "ipc"
WINDOW_LIMIT = 99
BATCH_QUIET_SECONDS = 1.0
BATCH_MAX_SECONDS = 5.0

_IS_WINDOWS = sys.platform == "win32"
_ENDPOINT_SUFFIX = "port" if _IS_WINDOWS else "sock"
_MAX_FRAME_BYTES = 1 << 20
_MAX_HELLO_BYTES = 512
_CONNECTION_SECONDS = 3.0
_AUTH_SECONDS = 1.0
_TOOL_SECONDS = 10.0
_KEY_LOCK_SECONDS = 3 * _TOOL_SECONDS + 5.0
_IO_TIMEOUT = 2.0
_LIVE_WINDOWS_SECONDS = 5.0
_MAX_CONNECTIONS = 8
_BATCH_ATTEMPTS = 3
_MAX_SOCKET_PATH_BYTES = 100
_PROTOCOL = "dipy-skyline-ipc-1"
_SERVER_LABEL = b"dipy-skyline server"
_CLIENT_LABEL = b"dipy-skyline client"
_REPLY_LABEL = b"dipy-skyline reply"
_KEY_BYTES = 32
_NONCE = re.compile(r"[0-9a-f]{64}")
_SID = re.compile(r"S-1-\d+(-\d+)+")
_MAX_PID = 1 << 31
_REQUEST_ERRORS = (OSError, ValueError, TypeError, KeyError, RecursionError)


def _read_key(key):
    """Read the authentication key, refusing a file that is not private.

    Parameters
    ----------
    key : Path
        Key file.

    Returns
    -------
    bytes
        The 32 bytes of the key.

    Raises
    ------
    OSError
        If the file is missing, is not owned by the user or readable by
        others (POSIX), or does not hold exactly 32 bytes.
    """
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
    fd = os.open(key, flags)
    try:
        if not _IS_WINDOWS:
            info = os.fstat(fd)
            if info.st_uid != os.getuid() or info.st_mode & 0o077:
                raise OSError(f"{key} is not a private file of yours")
        data = os.read(fd, _KEY_BYTES + 1)
    finally:
        os.close(fd)
    if len(data) != _KEY_BYTES:
        raise OSError(f"{key} does not hold a valid key")
    return data


def _windows_tool(name):
    """Locate a program of the Windows system folder.

    ``CreateProcess`` looks in the current folder before the system folder
    for a bare name, and the folder of a file opened from the file manager
    may hold a program planted by someone else.

    Parameters
    ----------
    name : str
        File name of the program, such as ``icacls.exe``.

    Returns
    -------
    str
        Absolute path of the program.
    """
    return str(Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / name)


def _run_windows_tool(name, *arguments):
    """Run a program of the Windows system folder without a window.

    Parameters
    ----------
    name : str
        File name of the program, such as ``icacls.exe``.
    *arguments : str
        Command-line arguments.

    Returns
    -------
    subprocess.CompletedProcess
        Finished process, whose output is text; bytes that do not decode,
        such as an account name in the console code page, are replaced.

    Raises
    ------
    OSError
        If the program cannot be run or takes more than ``_TOOL_SECONDS``.
    """
    try:
        return subprocess.run(
            [_windows_tool(name), *arguments],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=_TOOL_SECONDS,
            creationflags=subprocess.CREATE_NO_WINDOW,
            check=False,
        )
    except subprocess.TimeoutExpired as err:
        raise OSError(f"{name} did not finish in {_TOOL_SECONDS:g} seconds") from err


def _current_user_sid():
    """Find the security identifier of the current Windows user.

    Returns
    -------
    str
        The SID, such as ``S-1-5-21-...``, which unlike an account name does
        not depend on the domain or on the form of the name.

    Raises
    ------
    OSError
        If ``whoami`` cannot be run or does not print a SID.
    """
    result = _run_windows_tool("whoami.exe", "/user", "/nh")
    words = result.stdout.split()
    if result.returncode != 0 or not words or _SID.fullmatch(words[-1]) is None:
        raise OSError("cannot find the security identifier of the current user")
    return words[-1]


def _restrict_to_current_user(folder):
    """Remove every access to a Windows folder except the current user's.

    ``mkdir`` ignores its mode on Windows, so the folder would otherwise keep
    the permissions it inherits, which a redirected or shared profile can
    make wider than the user alone. Entries that were set on the folder
    itself are removed too.

    Parameters
    ----------
    folder : Path
        Existing folder.

    Raises
    ------
    OSError
        If the user cannot be identified, or ``icacls`` cannot be run, takes
        too long or refuses.
    """
    sid = _current_user_sid()
    for arguments in (("/reset",), ("/inheritance:r", "/grant:r", f"*{sid}:(OI)(CI)F")):
        result = _run_windows_tool("icacls.exe", str(folder), *arguments)
        if result.returncode != 0:
            raise OSError(f"icacls could not restrict {folder} to its user")


def _claim_when_free(name, ipc_dir, seconds):
    """Wait for the ownership of a name that another process holds briefly.

    Parameters
    ----------
    name : str
        Name to claim.
    ipc_dir : Path
        Folder of the endpoints.
    seconds : float
        Longest wait.

    Returns
    -------
    file object
        The open lock file, to give back with ``_release``.

    Raises
    ------
    OSError
        If the lock file cannot be opened or the name stays taken.
    """
    deadline = time.monotonic() + seconds
    while True:
        lock_file = _claim(name, ipc_dir)
        if lock_file is not None:
            return lock_file
        if time.monotonic() > deadline:
            raise OSError(f"{name} stayed locked by another launch")
        time.sleep(0.05)


def _create_key(ipc_dir, key):
    """Write a new key file, unless one appears first.

    Parameters
    ----------
    ipc_dir : Path
        Private folder of the endpoints.
    key : Path
        Key file to create.

    Returns
    -------
    bytes
        The key the file holds, which another process may have written.
    """
    fd, temporary = tempfile.mkstemp(dir=ipc_dir, prefix="authkey.", suffix=".tmp")
    temporary = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as key_file:
            key_file.write(secrets.token_bytes(_KEY_BYTES))
        try:
            os.link(temporary, key)
        except FileExistsError:
            pass
        except OSError:
            if not key.exists():
                os.replace(temporary, key)
    finally:
        temporary.unlink(missing_ok=True)
    return _read_key(key)


def ipc_authkey(*, ipc_dir=IPC_DIR):
    """Read the secret key of the window endpoints, creating it when absent.

    Parameters
    ----------
    ipc_dir : Path, optional
        Folder of the endpoints. Created, private to the user, when missing.

    Returns
    -------
    bytes
        The 32-byte key shared by the windows and launches of the user.

    Raises
    ------
    OSError
        If the folder or the key file is not private to the user, or cannot
        be read or written.

    Notes
    -----
    On Windows the folder is restricted to the user whenever the key is
    missing, under a lock that makes concurrent first launches restrict it
    once, before the key is written. No key is ever written to a folder that
    an earlier failure left open.
    """
    ipc_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not _IS_WINDOWS:
        fd = os.open(ipc_dir, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            if os.fstat(fd).st_uid != os.getuid():
                raise OSError(f"{ipc_dir} is not a folder owned by you")
            os.fchmod(fd, 0o700)
        finally:
            os.close(fd)

    key = ipc_dir / "authkey"
    try:
        return _read_key(key)
    except FileNotFoundError:
        pass

    if not _IS_WINDOWS:
        return _create_key(ipc_dir, key)
    lock_file = _claim_when_free("authkey", ipc_dir, _KEY_LOCK_SECONDS)
    try:
        try:
            return _read_key(key)
        except FileNotFoundError:
            _restrict_to_current_user(ipc_dir)
            return _create_key(ipc_dir, key)
    finally:
        _release(lock_file)


def _send_frame(sock, message):
    """Send a JSON object preceded by its length.

    Parameters
    ----------
    sock : socket.socket
        Connected socket.
    message : dict
        Object to send.

    Raises
    ------
    ValueError
        If the encoded message is larger than the frame limit.
    """
    data = json.dumps(message).encode()
    if len(data) > _MAX_FRAME_BYTES:
        raise ValueError("message is too large")
    sock.sendall(len(data).to_bytes(4, "big") + data)


def _recv_exactly(sock, count, deadline):
    """Receive exactly ``count`` bytes before a deadline.

    Parameters
    ----------
    sock : socket.socket
        Connected socket.
    count : int
        Number of bytes to read.
    deadline : float
        ``time.monotonic()`` value after which reading fails, however slowly
        the peer sends.

    Returns
    -------
    bytes
        The bytes read.

    Raises
    ------
    ConnectionError
        If the peer closes the connection first.
    TimeoutError
        If the deadline passes.
    """
    chunks = []
    remaining = count
    while remaining:
        left = deadline - time.monotonic()
        if left <= 0:
            raise TimeoutError("the peer took too long")
        sock.settimeout(left)
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("connection closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _recv_frame(sock, deadline, max_bytes):
    """Receive a JSON object sent by ``_send_frame``.

    Parameters
    ----------
    sock : socket.socket
        Connected socket.
    deadline : float
        ``time.monotonic()`` value after which reading fails.
    max_bytes : int
        Largest frame accepted.

    Returns
    -------
    dict
        The object received.

    Raises
    ------
    ValueError
        If the frame is larger than ``max_bytes`` or does not hold a JSON
        object.
    ConnectionError
        If the peer closes the connection mid-frame.
    TimeoutError
        If the deadline passes.
    """
    size = int.from_bytes(_recv_exactly(sock, 4, deadline), "big")
    if size > max_bytes:
        raise ValueError("message is too large")
    message = json.loads(_recv_exactly(sock, size, deadline))
    if not isinstance(message, dict):
        raise ValueError("message is not an object")
    return message


def _nonce():
    """Draw a random nonce.

    Returns
    -------
    str
        32 random bytes as 64 hexadecimal characters.
    """
    return secrets.token_hex(32)


def _check_nonce(value):
    """Check that a received nonce has the expected form.

    Parameters
    ----------
    value : object
        Value received from the peer.

    Raises
    ------
    ValueError
        If ``value`` is not 64 lowercase hexadecimal characters.
    """
    if not isinstance(value, str) or _NONCE.fullmatch(value) is None:
        raise ValueError("malformed nonce")


def _proof(key, label, first, second, *, body=""):
    """Compute the proof of knowing the key for one step of the handshake.

    Parameters
    ----------
    key : bytes
        Shared key.
    label : bytes
        Step label, so that a proof of one step is useless for another.
    first : str
        First nonce of the session.
    second : str
        Second nonce of the session.
    body : str, optional
        Text the proof vouches for, such as a request or a reply.

    Returns
    -------
    str
        Hexadecimal HMAC-SHA256 of the label, both nonces and the body.
    """
    message = label + first.encode() + second.encode() + body.encode()
    return hmac.new(key, message, "sha256").hexdigest()


def _proof_matches(received, expected):
    """Compare a received proof with the expected one in constant time.

    Parameters
    ----------
    received : object
        Value received from the peer.
    expected : str
        Proof computed locally.

    Returns
    -------
    bool
        True when ``received`` is the expected proof.

    Raises
    ------
    TypeError
        If ``received`` is a string that is not ASCII.
    """
    return isinstance(received, str) and hmac.compare_digest(received, expected)


def _serve_connection(conn, key, handler):
    """Run the server side of the handshake, then answer one request.

    Parameters
    ----------
    conn : socket.socket
        Accepted connection.
    key : bytes
        Shared key.
    handler : callable
        Function that takes the request and returns the reply.

    Raises
    ------
    ValueError
        If the client does not follow the protocol or does not know the key,
        or its request does not match its proof.
    TimeoutError
        If the client takes more than ``_AUTH_SECONDS`` to prove itself.
    """
    deadline = time.monotonic() + _AUTH_SECONDS
    hello = _recv_frame(conn, deadline, _MAX_HELLO_BYTES)
    if hello.get("protocol") != _PROTOCOL:
        raise ValueError("unknown protocol")
    client_nonce = hello.get("nonce")
    _check_nonce(client_nonce)
    server_nonce = _nonce()
    _send_frame(
        conn,
        {
            "nonce": server_nonce,
            "proof": _proof(key, _SERVER_LABEL, client_nonce, server_nonce),
        },
    )
    answer = _recv_frame(conn, deadline, _MAX_FRAME_BYTES)
    body = answer.get("body")
    if not isinstance(body, str):
        raise ValueError("request is not text")
    expected = _proof(key, _CLIENT_LABEL, server_nonce, client_nonce, body=body)
    if not _proof_matches(answer.get("proof"), expected):
        raise ValueError("client proof failed")
    reply = json.dumps(handler(json.loads(body)))
    _send_frame(
        conn,
        {
            "body": reply,
            "proof": _proof(key, _REPLY_LABEL, server_nonce, client_nonce, body=reply),
        },
    )


def _exchange(sock, key, request):
    """Run the client side of the handshake, then send one request.

    Parameters
    ----------
    sock : socket.socket
        Connected socket.
    key : bytes
        Shared key.
    request : dict
        Request to send once the server has proved it knows the key.

    Returns
    -------
    object
        The reply of the server, whose proof covers the request session.

    Raises
    ------
    ValueError
        If the server does not follow the protocol, does not know the key
        (no request is sent in that case), or its reply does not match its
        proof.
    TimeoutError
        If the exchange takes more than ``_CONNECTION_SECONDS`` in total.
    """
    deadline = time.monotonic() + _CONNECTION_SECONDS
    client_nonce = _nonce()
    _send_frame(sock, {"protocol": _PROTOCOL, "nonce": client_nonce})
    hello = _recv_frame(sock, deadline, _MAX_HELLO_BYTES)
    server_nonce = hello.get("nonce")
    _check_nonce(server_nonce)
    expected = _proof(key, _SERVER_LABEL, client_nonce, server_nonce)
    if not _proof_matches(hello.get("proof"), expected):
        raise ValueError("server proof failed")
    body = json.dumps(request)
    _send_frame(
        sock,
        {
            "body": body,
            "proof": _proof(key, _CLIENT_LABEL, server_nonce, client_nonce, body=body),
        },
    )
    answer = _recv_frame(sock, deadline, _MAX_FRAME_BYTES)
    reply = answer.get("body")
    if not isinstance(reply, str):
        raise ValueError("reply is not text")
    expected = _proof(key, _REPLY_LABEL, server_nonce, client_nonce, body=reply)
    if not _proof_matches(answer.get("proof"), expected):
        raise ValueError("reply proof failed")
    return json.loads(reply)


def _endpoint(name, ipc_dir):
    """Locate the file that publishes the endpoint called ``name``.

    Parameters
    ----------
    name : str
        ``window-N`` or ``launch``.
    ipc_dir : Path
        Folder of the endpoints.

    Returns
    -------
    Path
        The socket file (POSIX) or the file holding the port (Windows).
    """
    return ipc_dir / f"{name}.{_ENDPOINT_SUFFIX}"


def _claim(name, ipc_dir):
    """Take the ownership of a name, which lasts until the process ends.

    Parameters
    ----------
    name : str
        ``window-N`` or ``launch``.
    ipc_dir : Path
        Folder of the endpoints.

    Returns
    -------
    file object or None
        The open lock file, or None when another process owns the name.
    """
    lock_file = open(ipc_dir / f"{name}.lock", "a+b")  # noqa: SIM115
    try:
        if _IS_WINDOWS:
            lock_file.seek(0)
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        lock_file.close()
        return None
    return lock_file


def _release(lock_file):
    """Give up the ownership of a name.

    Parameters
    ----------
    lock_file : file object
        Lock file returned by ``_claim``.
    """
    try:
        if _IS_WINDOWS:
            lock_file.seek(0)
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
    finally:
        lock_file.close()


def exclusive_address_use(sock):
    """Stop other processes from binding the port of a socket, on Windows.

    On Windows a process that sets ``SO_REUSEADDR`` can bind a port that is
    already in use and receive its connections. ``SO_EXCLUSIVEADDRUSE`` has
    to be set before ``bind`` to prevent it.

    Parameters
    ----------
    sock : socket.socket
        Socket that has not been bound yet. Nothing happens on systems
        without that option.
    """
    if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)


def _bind(name, ipc_dir):
    """Create the listening socket of an endpoint and publish it.

    Parameters
    ----------
    name : str
        ``window-N`` or ``launch``, already claimed.
    ipc_dir : Path
        Folder of the endpoints.

    Returns
    -------
    socket.socket
        Listening socket whose ``accept`` times out every half second.

    Raises
    ------
    OSError
        If the socket cannot be created or its endpoint cannot be published,
        or the socket path is too long.
    """
    endpoint = _endpoint(name, ipc_dir)
    if _IS_WINDOWS:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    else:
        if len(os.fsencode(endpoint)) > _MAX_SOCKET_PATH_BYTES:
            raise OSError(f"{endpoint} is too long for a socket path")
        endpoint.unlink(missing_ok=True)
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        if _IS_WINDOWS:
            exclusive_address_use(sock)
            sock.bind(("127.0.0.1", 0))
        else:
            sock.bind(str(endpoint))
        sock.listen(_MAX_CONNECTIONS)
        sock.settimeout(0.5)
        if _IS_WINDOWS:
            fd, temporary = tempfile.mkstemp(
                dir=ipc_dir, prefix=f"{name}.", suffix=".tmp"
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as port_file:
                    port_file.write(str(sock.getsockname()[1]))
                os.replace(temporary, endpoint)
            except OSError:
                Path(temporary).unlink(missing_ok=True)
                raise
    except OSError:
        sock.close()
        raise
    return sock


def _connect(name, ipc_dir):
    """Connect to an endpoint.

    Parameters
    ----------
    name : str
        ``window-N`` or ``launch``.
    ipc_dir : Path
        Folder of the endpoints.

    Returns
    -------
    socket.socket
        Connected socket with an I/O timeout.

    Raises
    ------
    OSError
        If the endpoint does not exist or does not accept the connection.
    """
    if _IS_WINDOWS:
        try:
            port = int(_endpoint(name, ipc_dir).read_text(encoding="utf-8"))
        except ValueError as err:
            raise OSError(f"{name} has no valid port") from err
        if not 1 <= port <= 65535:
            raise OSError(f"{name} has no valid port")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        address = ("127.0.0.1", port)
    else:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        address = str(_endpoint(name, ipc_dir))
    sock.settimeout(_IO_TIMEOUT)
    try:
        sock.connect(address)
    except OSError:
        sock.close()
        raise
    return sock


class _Server:
    """Answer authenticated requests on a claimed endpoint, in threads.

    Parameters
    ----------
    name : str
        ``window-N`` or ``launch``.
    lock_file : file object
        Lock file that owns ``name``; released by ``close``.
    sock : socket.socket
        Listening socket made by ``_bind``.
    key : bytes
        Shared key.
    handler : callable
        Function that takes a request and returns the reply.
    ipc_dir : Path
        Folder of the endpoints.
    """

    def __init__(self, name, lock_file, sock, key, handler, ipc_dir):
        self._name = name
        self._lock_file = lock_file
        self._sock = sock
        self._key = key
        self._handler = handler
        self._ipc_dir = ipc_dir
        self._closing = threading.Event()
        self._slots = threading.BoundedSemaphore(_MAX_CONNECTIONS)
        self._thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._thread.start()

    def _accept_loop(self):
        """Accept connections until the server closes."""
        failing = False
        while not self._closing.is_set():
            try:
                conn, _ = self._sock.accept()
            except TimeoutError:
                failing = False
                continue
            except OSError as err:
                if self._closing.is_set():
                    return
                if not failing:
                    logger.warning(f"Skyline could not accept a connection: {err}")
                    failing = True
                time.sleep(0.1)
                continue
            failing = False
            if not self._slots.acquire(blocking=False):
                conn.close()
                continue
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _serve(self, conn):
        """Serve one connection, dropping it quietly on any failure.

        Parameters
        ----------
        conn : socket.socket
            Accepted connection with a timeout.
        """
        try:
            with conn:
                conn.settimeout(_IO_TIMEOUT)
                _serve_connection(conn, self._key, self._handler)
        except _REQUEST_ERRORS:
            pass
        finally:
            self._slots.release()

    def close(self):
        """Stop answering, remove the endpoint and give up the name.

        Calling it again does nothing.
        """
        if self._closing.is_set():
            return
        self._closing.set()
        self._sock.close()
        self._thread.join(1)
        endpoint = _endpoint(self._name, self._ipc_dir)
        try:
            endpoint.unlink(missing_ok=True)
        except OSError as err:
            logger.info(f"Could not remove {endpoint}: {err}")
        finally:
            _release(self._lock_file)


def _request(name, request, ipc_dir):
    """Send one request to an endpoint.

    Parameters
    ----------
    name : str
        ``window-N`` or ``launch``.
    request : dict
        Request to send.
    ipc_dir : Path
        Folder of the endpoints.

    Returns
    -------
    dict or None
        The reply, or None when the endpoint is missing, does not answer in
        time or does not know the key.
    """
    try:
        key = ipc_authkey(ipc_dir=ipc_dir)
        with _connect(name, ipc_dir) as sock:
            reply = _exchange(sock, key, request)
    except _REQUEST_ERRORS:
        return None
    return reply if isinstance(reply, dict) else None


@dataclass(frozen=True)
class WindowInfo:
    """A running Skyline window, as described by its own answer.

    Parameters
    ----------
    number : int
        Window number, which names its endpoint.
    pid : int
        Process id of the window.
    instance : str
        Random identifier of this window, which changes when another process
        takes the same number.
    last_active : float
        Time, as ``time.time()``, when the user last used the window or it
        last received files.
    """

    number: int
    pid: int
    instance: str
    last_active: float


class WindowServer:
    """Receive files for a Skyline window from other launches.

    Create it with ``WindowServer.start``.

    Parameters
    ----------
    number : int
        Window number, already claimed.
    lock_file : file object
        Lock file that owns the window number.
    sock : socket.socket
        Listening socket of the window.
    key : bytes
        Shared key.
    ipc_dir : Path
        Folder of the endpoints.
    """

    def __init__(self, number, lock_file, sock, key, ipc_dir):
        self.number = number
        self.instance = secrets.token_hex(16)
        self._last_active = time.time()
        self._lock = threading.Lock()
        self._closed = False
        self._received = queue.SimpleQueue()
        self._server = _Server(
            name=f"window-{number}",
            lock_file=lock_file,
            sock=sock,
            key=key,
            handler=self._handle,
            ipc_dir=ipc_dir,
        )

    @classmethod
    def start(cls, *, ipc_dir=IPC_DIR):
        """Take the lowest free window number and start listening.

        Parameters
        ----------
        ipc_dir : Path, optional
            Folder of the endpoints.

        Returns
        -------
        WindowServer or None
            The server, or None, after a warning, when the folder or the
            endpoint cannot be set up or all numbers are taken.
        """
        try:
            key = ipc_authkey(ipc_dir=ipc_dir)
            for number in range(1, WINDOW_LIMIT + 1):
                name = f"window-{number}"
                lock_file = _claim(name, ipc_dir)
                if lock_file is None:
                    continue
                try:
                    sock = _bind(name, ipc_dir)
                except OSError:
                    _release(lock_file)
                    raise
                return cls(
                    number=number,
                    lock_file=lock_file,
                    sock=sock,
                    key=key,
                    ipc_dir=ipc_dir,
                )
        except OSError as err:
            logger.warning(f"Skyline cannot receive files from other launches: {err}")
            return None
        logger.warning(
            "Skyline cannot receive files from other launches: "
            f"all {WINDOW_LIMIT} window numbers are taken."
        )
        return None

    def touch(self):
        """Record that the window is being used, now."""
        with self._lock:
            self._last_active = time.time()

    def receive(self):
        """Take the paths received since the last call, without waiting.

        Returns
        -------
        list of str
            Absolute paths of files to open, in the order received.
        """
        paths = []
        while True:
            try:
                paths.append(self._received.get_nowait())
            except queue.Empty:
                return paths

    def close(self):
        """Stop receiving files and release the window number.

        Files that arrive afterwards are refused, so that the launch that sent
        them opens a new window.
        """
        with self._lock:
            self._closed = True
        self._server.close()

    def _handle(self, request):
        """Answer a request.

        Parameters
        ----------
        request : object
            Request received after the handshake.

        Returns
        -------
        dict
            ``describe`` returns ``number``, ``pid``, ``instance`` and
            ``last_active``. ``open`` returns ``ok`` and, on refusal,
            ``error``; the files of a request are queued only when all of
            them pass ``checked_file`` and the window is not closed, and the
            window counts as used.
        """
        op = request.get("op") if isinstance(request, dict) else None
        if op == "describe":
            with self._lock:
                last_active = self._last_active
            return {
                "number": self.number,
                "pid": os.getpid(),
                "instance": self.instance,
                "last_active": last_active,
            }
        if op == "open":
            if request.get("instance") != self.instance:
                return {"ok": False, "error": "window changed"}
            paths = request.get("paths")
            if (
                not isinstance(paths, list)
                or not all(isinstance(path, str) for path in paths)
                or not 1 <= len(paths) <= MAX_FILES
            ):
                return {"ok": False, "error": f"expected 1 to {MAX_FILES} paths"}
            try:
                checked = [checked_file(path) for path in paths]
            except ValueError as err:
                return {"ok": False, "error": str(err)}
            with self._lock:
                if self._closed:
                    return {"ok": False, "error": "window closed"}
                for path in checked:
                    self._received.put(str(path))
                self._last_active = time.time()
            return {"ok": True}
        return {"ok": False, "error": "unknown request"}


def _finite_number(value):
    """Read a number received from a peer, refusing NaN and infinity.

    Parameters
    ----------
    value : object
        Value received from the peer.

    Returns
    -------
    float or None
        The value as a float, or None when it is not a finite number.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) else None


def _window_info(number, reply):
    """Check the description a window gave of itself.

    Parameters
    ----------
    number : int
        Window number whose endpoint answered.
    reply : dict
        Reply to a ``describe`` request.

    Returns
    -------
    WindowInfo or None
        The window, or None when the reply does not describe window
        ``number`` with a plausible process id, an identifier and a finite
        time.
    """
    pid = reply.get("pid")
    last_active = _finite_number(reply.get("last_active"))
    if (
        type(reply.get("number")) is not int
        or reply["number"] != number
        or type(pid) is not int
        or not 0 < pid < _MAX_PID
        or not isinstance(reply.get("instance"), str)
        or last_active is None
    ):
        return None
    return WindowInfo(
        number=number,
        pid=pid,
        instance=reply["instance"],
        last_active=last_active,
    )


def live_windows(*, ipc_dir=IPC_DIR):
    """List the Skyline windows that answer.

    Parameters
    ----------
    ipc_dir : Path, optional
        Folder of the endpoints.

    Returns
    -------
    list of WindowInfo
        Windows sorted by number. The endpoint file of a window number that
        no process owns any more is removed. Windows that were not asked
        within ``_LIVE_WINDOWS_SECONDS``, or whose lock file cannot be opened,
        are left out.
    """
    if not ipc_dir.is_dir():
        return []
    deadline = time.monotonic() + _LIVE_WINDOWS_SECONDS
    pattern = re.compile(rf"window-(\d+)\.{_ENDPOINT_SUFFIX}")
    windows = []
    for candidate in ipc_dir.glob(f"window-*.{_ENDPOINT_SUFFIX}"):
        match = pattern.fullmatch(candidate.name)
        if match is None or not 1 <= int(match.group(1)) <= WINDOW_LIMIT:
            continue
        if time.monotonic() > deadline:
            logger.warning("Skyline stopped asking windows: they took too long.")
            break
        number = int(match.group(1))
        name = f"window-{number}"
        try:
            lock_file = _claim(name, ipc_dir)
        except OSError as err:
            logger.info(f"Could not check {name}: {err}")
            continue
        if lock_file is not None:
            try:
                candidate.unlink(missing_ok=True)
            except OSError as err:
                logger.info(f"Could not remove {candidate}: {err}")
            finally:
                _release(lock_file)
            continue
        reply = _request(name, {"op": "describe"}, ipc_dir)
        window = None if reply is None else _window_info(number, reply)
        if window is not None:
            windows.append(window)
    return sorted(windows, key=lambda window: window.number)


def send_open(window, paths, *, ipc_dir=IPC_DIR):
    """Ask a window to open files.

    Parameters
    ----------
    window : WindowInfo
        Window returned by ``live_windows``.
    paths : sequence of str or Path
        Files to open, which the window checks again.
    ipc_dir : Path, optional
        Folder of the endpoints.

    Returns
    -------
    bool
        True when the window accepted all the files; False when it is gone,
        was replaced by another window, or refused a file.
    """
    reply = _request(
        f"window-{window.number}",
        {
            "op": "open",
            "instance": window.instance,
            "paths": [str(Path(path).absolute()) for path in paths],
        },
        ipc_dir,
    )
    return reply is not None and reply.get("ok") is True


class _Batch:
    """Paths gathered by the launch that owns the ``launch`` endpoint.

    Parameters
    ----------
    paths : sequence of str
        Paths of the launch that gathers.
    """

    def __init__(self, paths):
        self._lock = threading.Lock()
        self._paths = dict.fromkeys(paths)
        self._closed = False
        self._last_arrival = time.monotonic()

    def handle(self, request):
        """Add the paths of another launch.

        Parameters
        ----------
        request : object
            Request received after the handshake.

        Returns
        -------
        dict
            ``{"ok": True}`` when the paths were added. ``{"ok": False}``
            when the request is malformed, a path fails ``checked_file``, the
            batch is closed, or it would hold more than ``MAX_FILES`` paths.
        """
        if not isinstance(request, dict) or request.get("op") != "collect":
            return {"ok": False}
        paths = request.get("paths")
        if (
            not isinstance(paths, list)
            or not all(isinstance(path, str) for path in paths)
            or not 1 <= len(paths) <= MAX_FILES
        ):
            return {"ok": False}
        try:
            checked = [str(checked_file(path)) for path in paths]
        except ValueError:
            return {"ok": False}
        with self._lock:
            merged = {**self._paths, **dict.fromkeys(checked)}
            if self._closed or len(merged) > MAX_FILES:
                return {"ok": False}
            self._paths = merged
            self._last_arrival = time.monotonic()
        return {"ok": True}

    def wait_and_close(self, quiet, limit):
        """Wait until no launch has joined for a while, then stop accepting.

        Parameters
        ----------
        quiet : float
            Seconds without a new arrival after which the batch closes.
        limit : float
            Seconds since the start after which the batch closes anyway.

        Returns
        -------
        list of str
            Paths of this launch, then those received, without duplicates.
        """
        start = time.monotonic()
        while True:
            time.sleep(0.05)
            with self._lock:
                if time.monotonic() >= min(start + limit, self._last_arrival + quiet):
                    self._closed = True
                    return list(self._paths)


def collect_batch(
    paths, *, ipc_dir=IPC_DIR, quiet=BATCH_QUIET_SECONDS, limit=BATCH_MAX_SECONDS
):
    """Merge the files of launches that start together into one launch.

    The first launch owns the ``launch`` endpoint and gathers the files of the
    others, which hand theirs over and stop. Selecting several files in a
    file manager that starts one launcher for each is the typical case.

    Parameters
    ----------
    paths : sequence of str
        Absolute paths of the files this launch was asked to open.
    ipc_dir : Path, optional
        Folder of the endpoints.
    quiet : float, optional
        Seconds without another launch joining after which the gathering
        launch goes on.
    limit : float, optional
        Seconds after which the gathering launch goes on regardless.

    Returns
    -------
    list of str or None
        Paths to open, without duplicates, when this launch gathers or when
        it could not take part. None when its files were handed to the
        gathering launch, which then opens them.

    Raises
    ------
    ValueError
        If ``quiet`` or ``limit`` is negative or not finite.
    """
    for name, value in (("quiet", quiet), ("limit", limit)):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a finite number >= 0, got {value}")

    try:
        key = ipc_authkey(ipc_dir=ipc_dir)
    except OSError as err:
        logger.warning(f"Skyline cannot merge launches: {err}")
        return list(paths)

    for _ in range(_BATCH_ATTEMPTS):
        try:
            lock_file = _claim("launch", ipc_dir)
        except OSError as err:
            logger.warning(f"Skyline cannot merge launches: {err}")
            return list(paths)
        if lock_file is not None:
            batch = _Batch(paths)
            try:
                sock = _bind("launch", ipc_dir)
            except OSError as err:
                _release(lock_file)
                logger.warning(f"Skyline cannot merge launches: {err}")
                return list(paths)
            server = _Server(
                name="launch",
                lock_file=lock_file,
                sock=sock,
                key=key,
                handler=batch.handle,
                ipc_dir=ipc_dir,
            )
            try:
                return batch.wait_and_close(quiet=quiet, limit=limit)
            finally:
                server.close()
        reply = _request("launch", {"op": "collect", "paths": list(paths)}, ipc_dir)
        if reply is not None and reply.get("ok") is True:
            return None
        time.sleep(0.1)
    logger.warning(
        "Could not join the other Skyline launch; opening these files alone."
    )
    return list(paths)
