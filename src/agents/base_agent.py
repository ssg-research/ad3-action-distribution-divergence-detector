import os
import shutil

class base_agent:

    def __init__(self, env, args, device, name, victim):
        if victim:
            agent_mode = args.victim_agent_mode
        else:
            agent_mode = args.attacker_agent_mode

        if agent_mode == 'dqn' and args.use_dueling:
            model_name = 'ddqn'
        else:
            model_name = agent_mode
        
        if args.game_mode == "train":
            self.model_path = "./output/" + args.env_name + "/" + model_name + "/" + args.game_mode + "/"
        else:
            self.model_path = "./output/" + args.env_name + "/" + model_name + "/" + args.game_mode + "/" + name  
            self.defense_path = "./output/" + args.env_name + "/" + model_name + "/" + args.game_mode + "/" 

        if os.path.exists(self.model_path):
            shutil.rmtree(self.model_path, ignore_errors=True)
        os.makedirs(self.model_path)


    def save_run(self, score, step, run):
        #self.logger.add_score(score)
        pass

    def select_action(self, obs, explore_eps=0.5, rnn_hxs=None, masks=None, deterministic=False):
        pass

    def remember(self,obs, action, reward, next_obs, done):
        pass

    def update_agent(self, total_step, rollouts=None, advmask=None):
        pass