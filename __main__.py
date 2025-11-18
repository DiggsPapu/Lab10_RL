from src.trainning.dqn_train import train_dqn

    
def main():
    
    print("Starting training dqn...")
    
    train_dqn(episodes=1000)

if __name__ == "__main__":
    print("Running Lab10 package...")
    main()