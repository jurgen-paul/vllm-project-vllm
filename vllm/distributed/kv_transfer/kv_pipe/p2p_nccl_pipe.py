# SPDX-License-Identifier: Apache-2.0
import logging
import pickle
import socket
import threading
import typing
from typing import Optional

import torch
import zmq

from vllm.distributed.device_communicators.pynccl_wrapper import (
    NCCLLibrary, buffer_type, cudaStream_t, ncclComm_t, ncclDataTypeEnum)
from vllm.utils import current_stream

logger = logging.getLogger(__name__)


class P2pNcclPipe:

    def __init__(self,
                 local_rank: int,
                 hostname: str = "",
                 port: int = 0,
                 library_path: Optional[str] = None) -> None:
        self.local_rank = local_rank
        self.device = torch.device(f"cuda:{self.local_rank}")
        self.nccl = NCCLLibrary(library_path)

        if not hostname:
            hostname = socket.gethostname()
        if port == 0:
            raise ValueError("Port cannot be 0")
        self._hostname = hostname
        self._port = port
        self.local_address = f"{self._hostname}:{self._port}"

        self.context = zmq.Context()
        self.router_socket = self.context.socket(zmq.ROUTER)
        self.router_socket.bind(f"tcp://{self._hostname}:{self._port}")

        self.poller = zmq.Poller()
        self.poller.register(self.router_socket, zmq.POLLIN)

        self.store = {}  # tensor_id: torch.Tensor
        self.socks = {}  # remote_address: client socket
        self.comms = {}  # remote_address: ncclComm_t

        self._listener_thread = threading.Thread(
            target=self._listen_for_requests, daemon=True)
        self._listener_thread.start()

    def _create_connect(self, remote_address: typing.Optional[str] = None):
        assert remote_address is not None
        if remote_address not in self.socks:
            sock = self.context.socket(zmq.DEALER)
            sock.setsockopt_string(zmq.IDENTITY, self.local_address)
            sock.connect(f"tcp://{remote_address}")
            self.socks[remote_address] = sock
            if remote_address in self.comms:
                return sock, self.comms[remote_address]

            unique_id = self.nccl.ncclGetUniqueId()
            unique_id_obj = pickle.dumps(unique_id)
            data = {"cmd": "NEW", "unique_id": unique_id_obj}
            sock.send(pickle.dumps(data))

            with torch.cuda.device(self.device):
                rank = 0
                comm: ncclComm_t = self.nccl.ncclCommInitRank(
                    2, unique_id, rank)
                self.comms[remote_address] = (comm, rank)
                logger.info("ncclCommInitRank Success, %s 👉 %s, MyRank: %s",
                            self.local_address, remote_address, rank)

        return self.socks[remote_address], self.comms[remote_address]

    def send_tensor(
        self,
        tensor: torch.Tensor,
        tensor_id: str,
        remote_address: typing.Optional[str] = None,
    ):
        # logger.info(f"Send To {remote_address=}, {tensor_id=}, {tensor.shape=}, {tensor.dtype=}")
        if remote_address is None:
            self.store[tensor_id] = tensor
        else:
            if remote_address not in self.socks:
                self._create_connect(remote_address)

            sock = self.socks[remote_address]
            comm, rank = self.comms[remote_address]
            data = {
                "cmd": "PUT",
                "tensor_id": tensor_id,
                "shape": tensor.shape,
                "dtype": tensor.dtype
            }
            sock.send(pickle.dumps(data))

            response = sock.recv()
            if response == b"0" and tensor is not None:
                self._send(comm, tensor.to(self.device), rank ^ 1)
                logger.info(
                    "Send Tensor Success, %s 👉 %s, MyRank: %s, data: %s, tensor: %s",
                    self.local_address, remote_address, rank, data, tensor)

    def recv_tensor(
        self,
        tensor_id: str,
        remote_address: typing.Optional[str] = None,
    ) -> torch.Tensor:
        logger.info(f"Recv From {remote_address}, {tensor_id=}")

        if tensor_id in self.store:
            return self.store.pop(tensor_id)

        if remote_address is None:
            return None

        if remote_address not in self.socks:
            self._create_connect(remote_address)

        sock = self.socks[remote_address]
        comm, rank = self.comms[remote_address]

        data = {"cmd": "GET", "tensor_id": tensor_id}
        sock.send(pickle.dumps(data))

        message = sock.recv()
        data = pickle.loads(message)
        if data.ret == 0:
            tensor = torch.empty(data.shape,
                                 dtype=data.dtype,
                                 device=self.device)
            self._recv(comm, tensor, rank ^ 1)
            return tensor

        return None

    def _listen_for_requests(self):
        while True:
            socks = dict(self.poller.poll())
            if self.router_socket in socks:
                remote_address, message = self.router_socket.recv_multipart()
                data = pickle.loads(message)
                logger.debug("Received message from %s, data: %s",
                             remote_address.decode(), data)
                if data["cmd"] == "NEW":
                    unique_id = pickle.loads(data["unique_id"])
                    with torch.cuda.device(self.device):
                        rank = 1
                        comm: ncclComm_t = self.nccl.ncclCommInitRank(
                            2, unique_id, rank)
                        self.comms[remote_address] = (comm, rank)
                        logger.info(
                            "ncclCommInitRank Success, %s 👈 %s, MyRank: %s",
                            self.local_address, remote_address.decode(), rank)
                elif data["cmd"] == "PUT":
                    tensor_id = data["tensor_id"]
                    self.router_socket.send_multipart([remote_address, b"0"])
                    tensor = torch.empty(data["shape"],
                                         dtype=data["dtype"],
                                         device=self.device)
                    comm, rank = self.comms[remote_address]
                    self._recv(comm, tensor, rank ^ 1)
                    self.store[tensor_id] = tensor
                    logger.info(
                        "Recv Tensor Success, %s 👈 %s, MyRank: %s, data: %s, tensor: %s",
                        self.local_address, remote_address.decode(), rank,
                        data, tensor)
                elif data["cmd"] == "GET":
                    tensor_id = data["tensor_id"]
                    if tensor_id in self.store:
                        data = {
                            "ret": 0,
                            "shape": self.store[tensor_id].shape,
                            "dtype": self.store[tensor_id].dtype
                        }
                    else:
                        data = {"ret": 1}
                    self.router_socket.send_multipart(
                        [remote_address, pickle.dumps(data)])
                    if data["ret"] == 0:
                        self._send(comm, self.store[tensor_id].to(self.device),
                                   rank ^ 1)
                else:
                    logger.info("Bug, Received message from %s, data: %s",
                                remote_address, data)

    def _send(self, comm, tensor: torch.Tensor, dst: int, stream=None):
        assert tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {tensor.device}")
        if stream is None:
            stream = current_stream()
        self.nccl.ncclSend(buffer_type(tensor.data_ptr()), tensor.numel(),
                           ncclDataTypeEnum.from_torch(tensor.dtype), dst,
                           comm, cudaStream_t(stream.cuda_stream))

    def _recv(self, comm, tensor: torch.Tensor, src: int, stream=None):
        assert tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {tensor.device}")
        if stream is None:
            stream = current_stream()
        self.nccl.ncclRecv(buffer_type(tensor.data_ptr()), tensor.numel(),
                           ncclDataTypeEnum.from_torch(tensor.dtype), src,
                           comm, cudaStream_t(stream.cuda_stream))
