NPCs = {
    # Snakes
    21: 3,
    31: 8,
    47: 30,
    52: 23,
    76: 41,
    81: 62,
    88: 67,
    98: 12,

    # Ladders
    4: 75,
    5: 15,
    19: 41,
    28: 50,
    35: 96,
    44: 82,
    58: 94,
    59: 95,
    70: 91,
}


class SnakeLadder:

    def __init__(self, start = 0, end = 100, NPCs = NPCs, action_space = 6, reward_origin = 10, max_steps = 15):
        self.start = start
        self.end = end
        self.NPCs = NPCs
        self.action_space = action_space
        self.state_space = end - start
        self.reward_origin = reward_origin
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        self.state = self.start
        self.turns = 0
        return self.state

    def step(self, action, render = False):
        self.turns += 1
        went_to = self.state + action + 1  # +1 because action is 0-indexed

        if self.turns >= self.max_steps:
            if render: self.render("Max steps reached")
            return went_to if went_to < self.end else self.state, True, (-self.max_steps)/1

        if self.NPCs.get(went_to):
            went_to = self.NPCs[went_to]
        if went_to > self.end:
            if render: self.render("Exceeded end")
            return self.state, False, 0

        self.state = went_to

        if went_to == self.end:
            if render: self.render("Winner!")
            return went_to, True, (self.reward_origin - self.turns)/1

        if render: self.render()
        return went_to, self.turns >= self.max_steps, 0

    def render(self, message=None):
        print(f"Current state: {self.state}", f", Message: {message}" if message else "")
        print(f"In Step: {self.turns} of {self.max_steps} steps")