import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="HPMODEL")
    parser.add_argument("--num_episodes", type=int, default=100000)
    parser.add_argument("--network_choice", type=str, default='RNN_LSTM_onlyLastHidden')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seq", type=str, default='HHHPPHPHPHPPHPHPHPPH')
    parser.add_argument("--agent_choice", type=str, default='dqn')
    parser.add_argument("--buffer", type=str, default='random')
    parser.add_argument("--update_timestep", type=int, default=300)
    parser.add_argument("--K_epochs", type=int, default=150)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    print(parse_args())
