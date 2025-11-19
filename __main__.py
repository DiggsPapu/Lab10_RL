from src.trainning.dqn_train import train_dqn
from src.trainning.a2c import train_a2c

    
def main():
    
    # print("Starting training dqn...")
    # train_dqn(episodes=5000)
    print("Starting training a2c...")
    model = train_a2c(episodes=1000, n_steps=5, gamma=0.99, lr=7e-4,
                      entropy_coef=0.01, value_coef=0.5, save_path="a2c_galaxian.pt")
    print("Training finished. Model saved to a2c_galaxian.pt")

if __name__ == "__main__":
    print("Running Lab10 package...")
    main()