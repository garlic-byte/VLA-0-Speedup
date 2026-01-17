from robot.policy import PolicyClient
import numpy as np


policy = PolicyClient(host="localhost", port=8000) # Connect to the policy server
if not policy.ping(): # Verify connection
    raise RuntimeError("Cannot connect to policy server!")
obs = {
        "video":
            {
                "image": np.zeros((480, 640, 3), dtype=np.uint8),
                "wrist_image": np.zeros((480, 640, 3), dtype=np.uint8),
            },
        "language": "put the white mug on the left plate and put the yellow and white mug on the right plate",
}
action = policy.get_action(obs) # Run inference

print(action)
