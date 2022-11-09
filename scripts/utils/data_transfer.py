import io
import torch
import os

def torch_to_bytes(t: torch.Tensor) -> bytes:
    buff = io.BytesIO()
    torch.save(t, buff)
    buff.seek(0)
    return buff.read()

def bytes_to_torch(b: bytes) -> torch.Tensor:
    buff = io.BytesIO(b)
    return torch.load(buff)

def check_server(server_address):
    try:
        os.unlink(server_address)
    except OSError:
        if os.path.exists(server_address):
            raise