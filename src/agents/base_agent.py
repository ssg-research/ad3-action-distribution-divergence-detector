import datetime
from rl_utils.logger import Logger

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
            self.logger_path = "./output/logs/" + args.env_name + "/" + model_name + "/" + args.game_mode +  "_" + self._get_date() + "/"
            self.model_path = "./output/nets/" + args.env_name + "/" + model_name + "/" + args.game_mode + "/"
            self.logger = Logger(args.env_name + " " + model_name, self.logger_path, self.model_path) 
        else:
            self.model_path = "./output/nets/" + args.env_name + "/" + model_name + "/" + args.game_mode + "/" + name  
            self.defense_path = "./output/nets/" + args.env_name + "/" + model_name + "/" + args.game_mode + "/" 

    def save_run(self, score, step, run):
        self.logger.add_score(score)
        #self.logger.add_step(step)
        #self.logger.add_run(run)
        pass

    def select_action(self, obs, explore_eps=0.5, rnn_hxs=None, masks=None, deterministic=False):
        pass

    def remember(self,obs, action, reward, next_obs, done):
        pass

    def update_agent(self, total_step, rollouts=None, advmask=None):
        pass

    def _get_date(self):
        return str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))

