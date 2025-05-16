import zmq
import torch
import torch.nn.functional as F
import pickle
import time

NUM_CLIENTS = 3
DISTILL_TEMPERATURE = 3
PORT = 5555

context = zmq.Context()
socket = context.socket(zmq.ROUTER)  # use ROUTER for multi-client handling
socket.bind(f"tcp://*:{PORT}")

print("Server started and waiting for clients...")

client_logit_buffer = {}
current_round = 1

while True:
    identity, _, data = socket.recv_multipart()
    payload = pickle.loads(data)
    client_id = payload["client_id"]
    logits = payload["logits"]
    round_num = payload["round"]

    print(f"[Round {round_num}] Received logits from Client {client_id}")

    if round_num not in client_logit_buffer:
        client_logit_buffer[round_num] = {}
    
    client_logit_buffer[round_num][client_id] = logits

    # Wait for all clients
    if len(client_logit_buffer[round_num]) < NUM_CLIENTS:
        continue

    # Compute average logits
    all_logits = list(client_logit_buffer[round_num].values())
    stacked_logits = torch.stack(all_logits)
    avg_logits = stacked_logits.mean(dim=0)

    # Send averaged logits back to each client
    teacher_prob = F.softmax(avg_logits / DISTILL_TEMPERATURE, dim=1)
    for cid in range(1, NUM_CLIENTS + 1):
        msg = {
            "teacher_prob": teacher_prob,
            "round": round_num
        }
        socket.send_multipart([f"client{cid}".encode(), b"", pickle.dumps(msg)])

    print(f"[Round {round_num}] Sent teacher logits to all clients\n")
    current_round += 1


