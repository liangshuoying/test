import struct

from twisted.internet.protocol import DatagramProtocol
#~ from twisted.internet.defer import inlineCallbacks, Deferred, returnValue
from twisted.internet import reactor


class Panda3dUDP(DatagramProtocol):
    
    def __init__(self, p3d_step, action_fmt, obs_pixel=0, act_disc=0):
        """ @p3d_step: 控制p3d单步step
        """
        self.p3d_step = p3d_step
        self.obs_pixel = obs_pixel
        self.act_disc = act_disc
        
        self.game_state = None
        self.game_done = 0
        self.game_action = {}
        
        self.action_names = action_fmt.get('name', ['turn', 'accel', 'fire'] )
        self.action_format = action_fmt.get('fmt', 'I3f')
    
    def datagramReceived(self, data, addr):
        """ 接收gym接口发送的udp数据
        """
        repeat_action = self.parse_game_action(data)
        # Exit Flag
        if repeat_action==99:
            reactor.stop()
        # Restart Game And Clear All States
        elif repeat_action==100:
            repeat_action = 0
            self.game_action.update(restart=True)
            self.p3d_step()
            self.game_action.update(restart=False)
        # Exit End
        self.p3d_step()
        for i in range(repeat_action):
            self.p3d_step()
            if self.game_done:
                break
        if self.game_state is not None:
            self.transport.write(self.game_state, addr)
    
    def set_game_state(self, fmt, obs_state, done):
        """ 游戏状态输出 """
        self.game_state = struct.pack(fmt, *obs_state)
        self.game_done = done
    
    def set_game_state_pixel(self, fmt, obs, state, done):
        """ 游戏状态输出
        @obs: ndarray.tobytes
        """
        self.game_state = struct.pack(fmt, obs, *state)
        self.game_done = done
    
    def parse_game_action(self, data):
        """ 游戏动作输入 'I3f'
        @data: 一个整数int(动作重复)
               三个浮点float(动作值)
        """
        data = struct.unpack(self.action_format, data)
        self.game_action = dict( zip(self.action_names, data[1:]) )
        repeat_action = data[0]
        return repeat_action
    
    def get_game_action(self):
        """ P3D引擎获取动作值 """
        return self.game_action


def main(port=8888):
    reactor.listenUDP(port, Panda3dUDP())
    reactor.run()


if __name__ == '__main__':
    main()

