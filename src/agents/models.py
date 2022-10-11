
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

# the convolution layer of deepmind
class deepmind(nn.Module):
    def __init__(self):
        super(deepmind, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 32, 3, stride=1)
        
        # start to do the init...
        nn.init.orthogonal_(self.conv1.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.orthogonal_(self.conv2.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.orthogonal_(self.conv3.weight.data, gain=nn.init.calculate_gain('relu'))
        # init the bias...
        nn.init.constant_(self.conv1.bias.data, 0)
        nn.init.constant_(self.conv2.bias.data, 0)
        nn.init.constant_(self.conv3.bias.data, 0)

        self.gradients = None

    def activations_hook(self, grad):
        self.gradients = grad
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # register the hook
        #h = x.register_hook(self.activations_hook)
        x = x.view(-1, 32 * 7 * 7)
        return x

    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return F.relu(self.conv3(x))

# in the initial, just the nature CNN
class DQNnet(nn.Module):
    def __init__(self, num_actions, use_dueling=False):
        super(DQNnet, self).__init__()
        # if use the dueling network
        self.use_dueling = use_dueling
        # define the network
        self.cnn_layer = deepmind()
        # if not use dueling
        if not self.use_dueling:
            self.fc1 = nn.Linear(32 * 7 * 7, 256)
            self.action_value = nn.Linear(256, num_actions)
        else:
            # the layer for dueling network architecture
            self.action_fc = nn.Linear(32 * 7 * 7, 256)
            self.state_value_fc = nn.Linear(32 * 7 * 7, 256)
            self.action_value = nn.Linear(256, num_actions)
            self.state_value = nn.Linear(256, 1)

    def forward(self, inputs):
        x = self.cnn_layer(inputs / 255.0)
        if not self.use_dueling:
            x = F.relu(self.fc1(x))
            action_value_out = self.action_value(x)
        else:
            # get the action value
            action_fc = F.relu(self.action_fc(x))
            action_value = self.action_value(action_fc)
            # get the state value
            state_value_fc = F.relu(self.state_value_fc(x))
            state_value = self.state_value(state_value_fc)
            # action value mean
            action_value_mean = torch.mean(action_value, dim=1, keepdim=True)
            action_value_center = action_value - action_value_mean
            # Q = V + A
            action_value_out = state_value + action_value_center
        return action_value_out

# Categorical
class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)

class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()

        init_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            gain=0.01)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedCategorical(logits=x)

class A2Cnet(nn.Module):
    def __init__(self, num_actions, recurrent=False, hidden_size=512):
        super(A2Cnet, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.num_outputs = num_actions
        self._hidden_size = hidden_size

        self.main = nn.Sequential(
            init_(nn.Conv2d(4, 32, 8, stride=4)), nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)), nn.ReLU(), Flatten(),
            init_(nn.Linear(32 * 7 * 7, hidden_size)), nn.ReLU())

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

        self.dist = Categorical(self._hidden_size, self.num_outputs)
    
    @property
    def output_size(self):
        return self._hidden_size

    def forward(self, inputs, rnn_hxs, masks):
        x = self.main(inputs / 255.0)
        return self.critic_linear(x), x, rnn_hxs
