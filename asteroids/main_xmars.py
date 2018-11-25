#!/usr/bin/env python



from direct.showbase.ShowBase import ShowBase
from panda3d.core import TextNode, TransparencyAttrib
from panda3d.core import LPoint3, LVector3, LVector4, Vec3
from panda3d.core import ClockObject
from direct.gui.OnscreenText import OnscreenText
from direct.task.Task import Task
from math import sin, cos, pi
from random import randint, choice, random
from direct.interval.MetaInterval import Sequence
from direct.interval.FunctionInterval import Wait, Func
import sys

from direct.directtools.DirectGeometry import LineNodePath
from panda3d.core import loadPrcFileData
import numpy as np
import cv2
#~ from PIL import Image

from crafts import Ship
from config import *

import argparse

parser = argparse.ArgumentParser(description='Panda3d')
parser.add_argument('--port', type=int, default=8880, help='Server Port')
parser.add_argument('--render', type=int, default=0, help='Render Me')
args = parser.parse_args()

#~ np.random.seed(321)

if DOG_FIGHT==2:
    WIN_W, WIN_H = 640, 640
else:
    WIN_W, WIN_H = 512, 512
if not args.render:
    WIN_W, WIN_H = 400, 400
    loadPrcFileData("", "window-type offscreen")
#~ loadPrcFileData("", "load-display p3tinydisplay")
#~ loadPrcFileData("", "framebuffer-hardware 0")
#~ loadPrcFileData("", "framebuffer-software 1")
loadPrcFileData('', 'win-size %d %d'%(WIN_W, WIN_H))
#~ loadPrcFileData('', 'show-frame-rate-meter 1')


def P3DCreateAxes(length=0.5, arrowAngle=25, arrowLength=0.05):
    nodePath = LineNodePath()
    nodePath.reparentTo(render2d)
    #~ nodePath.drawArrow2d(Vec3(0, 0, 0), Vec3(0.5, 0, 0.5), 30, 0.1)
    nodePath.drawArrow2d(Vec3(0, 0, 0), Vec3(0., 0, length), arrowAngle, arrowLength)
    nodePath.create()
    return nodePath

# This helps reduce the amount of code used by loading objects, since all of
# the objects are pretty much the same.
def loadObject(tex=None, pos=LPoint3(0, 0), depth=SPRITE_POS, scale=1,
               transparency=True):
    # Every object uses the plane model and is parented to the camera
    # so that it faces the screen.
    obj = loader.loadModel("models/plane")
    obj.reparentTo(camera)

    # Set the initial position and scale.
    obj.setPos(pos.getX(), depth, pos.getY())
    obj.setScale(scale)

    # This tells Panda not to worry about the order that things are drawn in
    # (ie. disable Z-testing).  This prevents an effect known as Z-fighting.
    obj.setBin("unsorted", 0)
    obj.setDepthTest(False)

    if transparency:
        # Enable transparency blending.
        obj.setTransparency(TransparencyAttrib.MAlpha)

    if tex:
        # Load and set the requested texture.
        tex = loader.loadTexture("textures/" + tex)
        obj.setTexture(tex, 1)

    return obj


# Macro-like function used to reduce the amount to code needed to create the
# on screen instructions
def genLabelText(text, i):
    return OnscreenText(text=text, parent=base.a2dTopLeft, pos=(0.07, -.06 * i - 0.1),
                        fg=(1, 1, 1, 1), align=TextNode.ALeft, shadow=(0, 0, 0, 0.5), scale=.05)


class AsteroidsDemo(ShowBase):

    def __init__(self, p3d_proto=None, obs_pixel=0, act_disc=0):
        # Initialize the ShowBase class from which we inherit, which will
        # create a window and set up everything we need for rendering into it.
        ShowBase.__init__(self)
        
        self.p3d_proto = p3d_proto
        self.obs_pixel = obs_pixel
        self.act_disc = act_disc
        
        globalClock.setFrameRate(30)
        # MNormal:0 MNoneRealTime:1 MForced:2 MLimited:5
        #~ globalClock.setMode(0)
        globalClock.setMode(1)
        
        self.exitFunc=on_exit
        
        if SHOW_ARROW:
            self.ship_arrow = P3DCreateAxes(length=GUN_RANGE/SCREEN_X)
            self.ship_arrow1 = P3DCreateAxes(length=GUN_RANGE/SCREEN_X)
        #~ self.ship_arrow1 = P3DCreateAxes(length=AVOID_DIST/SCREEN_X)
        
        #~ self.ast_arrow = P3DCreateAxes(length=GUN_RANGE/SCREEN_X)
        #~ self.ast_arrow1 = P3DCreateAxes(length=AVOID_DIST/SCREEN_X)
        
        self.bullets_text = OnscreenText(text="bullets 100:100",
                                  parent=base.a2dBottomRight, scale=.1,
                                  align=TextNode.ALeft, pos=(-1.99, 1.8),
                                  fg=(0, 1, 0, 1), shadow=(0, 0, 0, 0.5))
        
        self.life_text = OnscreenText(text="life 10:10",
                                  parent=base.a2dBottomRight, scale=.1,
                                  align=TextNode.ALeft, pos=(-1.99, 1.7),
                                  fg=(1, 0, 0, 1), shadow=(0, 0, 0, 0.5))
        
        self.debug_text = OnscreenText(text="DBG --",
                                  parent=base.a2dBottomRight, scale=.1,
                                  align=TextNode.ALeft, pos=(-1.99, 1.9),
                                  fg=(1, 1, 0, 1), shadow=(0, 0, 0, 0.5))
        
        self.text_agent1 = OnscreenText(text="A1",
                                  parent=base.a2dBottomRight, scale=.1,
                                  align=TextNode.ALeft, pos=(-1, 1),
                                  fg=(0, 1, 0, 1), shadow=(0, 0, 0, 0.5))
        
        # Disable default mouse-based camera control.  This is a method on the
        # ShowBase class from which we inherit.
        self.disableMouse()

        # Load the background starfield.
        self.setBackgroundColor((0, 0, 0, 1))
        self.bg = loadObject("stars.jpg", scale=146, depth=200,
                             transparency=False)

        # Load the ship and set its initial velocity.
        #~ self.ship = loadObject("ship.png", scale=2)
        #~ self.setVelocity(self.ship, LVector3.zero())
        self.ship1 = Ship(self)
        self.ship2 = Ship(self)

        # A dictionary of what keys are currently being pressed
        # The key events update this list, and our task will query it as input
        self.keys = {"turnLeft": 0, "turnRight": 0,
                     "accel": 0, "fire": 0}

        #~ self.accept("escape", sys.exit)  # Escape quits
        self.accept("escape", on_exit)  # Escape quits
        # Other keys events set the appropriate value in our key dictionary
        self.accept("a",     self.setKey, ["turnLeft", 1])
        self.accept("a-up",  self.setKey, ["turnLeft", 0])
        self.accept("d",    self.setKey, ["turnRight", 1])
        self.accept("d-up", self.setKey, ["turnRight", 0])
        self.accept("w",       self.setKey, ["accel", 1])
        self.accept("w-up",    self.setKey, ["accel", 0])
        
        self.accept("arrow_left",     self.setKey, ["turnLeft", 1])
        self.accept("arrow_left-up",  self.setKey, ["turnLeft", 0])
        self.accept("arrow_right",    self.setKey, ["turnRight", 1])
        self.accept("arrow_right-up", self.setKey, ["turnRight", 0])
        self.accept("arrow_up",       self.setKey, ["accel", 1])
        self.accept("arrow_up-up",    self.setKey, ["accel", 0])
        
        self.accept("space",          self.setKey, ["fire", 1])
        self.accept("space-up",          self.setKey, ["fire", 0])

        self.gameTask = taskMgr.add(self.gameLoop, "gameLoop")

        # Stores the time at which the next bullet may be fired.
        self.nextBullet = 0.0
        
        # This list will stored fired bullets.
        self.bullets = []
        
        self.reset_hud()

        # Complete initialization by spawning the asteroids.
        self.spawnAsteroids()
    
    def reset_hud(self):
        self.rounds = 0
        self.count_win = 50
        self.draws = [0] *self.count_win
        self.wins1 = [0] *self.count_win
        self.wins2 = [0] *self.count_win
        self.wins1s = 0
        self.wins2s = 0
    
    # As described earlier, this simply sets a key in the self.keys dictionary
    # to the given value.
    def setKey(self, key, val):
        self.keys[key] = val

    def setVelocity(self, obj, val):
        obj.setPythonTag("velocity", val)

    def getVelocity(self, obj):
        return obj.getPythonTag("velocity")

    def setExpires(self, obj, val):
        obj.setPythonTag("expires", val)

    def getExpires(self, obj):
        return obj.getPythonTag("expires")
    
    def setAttacker(self, obj, val):
        obj.setPythonTag("attacker", val)

    def getAttacker(self, obj):
        return obj.getPythonTag("attacker")

    def spawnAsteroids(self):
        # Control variable for if the ship is alive
        self.alive = True
        self.asteroids = []  # List that will contain our asteroids

        for i in range(AST_NUM):
            # This loads an asteroid. The texture chosen is random
            # from "asteroid1.png" to "asteroid3.png".
            asteroid = loadObject("asteroid%d.png" % (randint(1, 3)),
                                  scale=AST_INIT_SCALE)
            self.asteroids.append(asteroid)

            # This is kind of a hack, but it keeps the asteroids from spawning
            # near the player.  It creates the list (-20, -19 ... -5, 5, 6, 7,
            # ... 20) and chooses a value from it. Since the player starts at 0
            # and this list doesn't contain anything from -4 to 4, it won't be
            # close to the player.
            asteroid.setX(choice(tuple(range(-SCREEN_X, -5)) + tuple(range(5, SCREEN_X))))
            # Same thing for Y
            asteroid.setZ(choice(tuple(range(-SCREEN_Y, -5)) + tuple(range(5, SCREEN_Y))))

            # Heading is a random angle in radians
            heading = random() * 2 * pi

            # Converts the heading to a vector and multiplies it by speed to
            # get a velocity vector
            v = LVector3(sin(heading), 0, cos(heading)) * AST_INIT_VEL
            self.setVelocity(self.asteroids[i], v)

    # This is our main task function, which does all of the per-frame
    # processing.  It takes in self like all functions in a class, and task,
    # the task object returned by taskMgr.
    def gameLoop(self, task):
        # Get the time elapsed since the next frame.  We need this for our
        # distance and velocity calculations.
        dt = globalClock.getDt()
        SHOW_ARROW=False

        if not self.alive:
            return Task.cont
        
        # Net Update Action
        action = self.net_update_action()
        a_turn = action.get('turn', 0)
        a_accel = action.get('accel', -1)
        a_fire = action.get('fire', -1)
        a_turn2 = action.get('turn2', 0)
        a_accel2 = action.get('accel2', -1)
        a_fire2 = action.get('fire2', -1)
        # Net Update End
        
        # Reset By Agent Begin
        if action.get('restart', False):
            self.resetGame(restart=1)
            return Task.cont
        # Reset End
        
        # update ship position
        self.ship1.updateShip(dt, a_turn, a_accel)
        if ONE_SHIP_ONLY:
            self.ship1.hide()
            self.ship_arrow.hide()
            self.text_agent1.hide()
        else:
            self.ship1.updateFire(task, a_fire)
        # AI vs Player Begin
        if DOG_FIGHT==2:
            a_turn2 = self.keys["turnRight"] - self.keys["turnLeft"]
            a_accel2 = self.keys["accel"]
            a_fire2 = self.keys["fire"]
            # slower turn_rate for human
            turn_ra=TURN_RATE*0.66
        # AVP End
        else:
            turn_ra = TURN_RATE
        self.ship2.updateShip(dt, a_turn2, a_accel2, turn_ra=turn_ra)
        self.ship2.updateFire(task, a_fire2)
        
        # check crash to each other
        if DOG_FIGHT:
            self.ship1.checkCrash(self.ship2.ship)
            self.ship2.checkCrash(self.ship1.ship)
        # end
        
        # update asteroids
        for obj in self.asteroids:
            self.updatePos(obj, dt)
        
        # update bullets
        newBulletArray = []
        for obj in self.bullets:
            self.updatePos(obj, dt)  # Update the bullet
            # Bullets have an experation time (see definition of fire)
            if self.getExpires(obj) > task.time:
                newBulletArray.append(obj)
            else:
                obj.removeNode()  # Otherwise, remove it from the scene.
        # Set the bullet array to be the newly updated array
        self.bullets = newBulletArray

        for bullet in self.bullets:
            # check ship1 vs ship2
            if DOG_FIGHT:
                if self.ship1.checkHit(bullet):
                    if self.ship1.alive:
                        self.ship2.game_score += 10
                    else:
                        self.ship2.game_score += 100
                        self.wins2s += self.ship2.alive
                elif self.ship2.checkHit(bullet):
                    if self.ship2.alive:
                        self.ship1.game_score += 10
                    else:
                        self.ship1.game_score += 100
                        self.wins1s += self.ship1.alive
            # end
            for i in range(len(self.asteroids) - 1, -1, -1):
                asteroid = self.asteroids[i]
                if ((bullet.getPos() - asteroid.getPos()).lengthSquared() <
                    (((bullet.getScale().getX() + asteroid.getScale().getX())
                      * .5) ** 2)):
                    # Schedule the bullet for removal
                    self.setExpires(bullet, 0)
                    self.asteroidHit(i)      # Handle the hit
                    # Scoring Begin
                    if self.getAttacker(bullet)==id(self.ship1):
                        self.ship1.game_score += 10
                    else:
                        self.ship2.game_score += 10
                    # Scoring End
        
        # DEBUG BEGIN
        wrn = self.count_win / 100.
        wr1 = sum(self.wins1) / wrn
        wr2 = sum(self.wins2) / wrn
        drr = sum(self.draws) / wrn
        self.debug_text.setText("r%d: A1 %d:%d, wr%% %d:%d, dr %d%%, crash %d:%d"%(self.rounds, 
                                    self.wins1s, self.wins2s, wr1, wr2, drr, self.ship1.crash, self.ship2.crash) )
        # DEBUG END
        
        # If the player has successfully destroyed all asteroids, respawn them
        if len(self.asteroids) == 0:
            self.spawnAsteroids()
        
        # Net Update State
        game_over = self.net_update_state(done=0)
        # Net Update End
        if game_over:
            # win rate begin
            self.wins1.append( self.ship1.alive )
            self.wins1.pop(0)
            self.wins2.append( self.ship2.alive )
            self.wins2.pop(0)
            self.draws.append(0 if self.ship1.alive+self.ship2.alive else 1)
            self.draws.pop(0)
            # win rate end
            self.resetGame()
        
        return Task.cont    # Since every return is Task.cont, the task will
        # continue indefinitely
    
    # Net Update Begin
    
    def resetGame(self, wait=0, restart=0):
        self.alive = False
        for i in self.asteroids + self.bullets:
            i.removeNode()
        self.bullets = []          # Clear the bullet list
        self.rounds += 1
        
        if restart:
            self.reset_hud()
        
        Sequence(Wait(wait),          # Wait 2 seconds
                 Func(self.ship1.reset, restart),     # Show the ship
                 Func(self.ship2.reset, restart),     # Show the ship
                 Func(self.spawnAsteroids)).start()  # Remake asteroids
    
    def getFrame(self, render=0):
        """ 84x84 1 channel """
        if render:
            self.graphicsEngine.renderFrame()
        dr = self.camNode.getDisplayRegion(0)
        img = dr.getScreenshot().getRamImageAs('G')
        img = np.array(img, dtype=np.uint8).reshape((WIN_H, WIN_W))
        img = cv2.resize(img, (84, 84), interpolation=cv2.INTER_AREA)
        return img
    
    def net_update_state(self, done=0):
        """ 游戏状态, 输出到udp """
        ship1_state, ship1_obs, ship1_score, ast_critical1 = self.ship1.getState(self.ship2.ship, self.asteroids, invincible=ONE_SHIP_ONLY)
        ship2_state, ship2_obs, ship2_score, ast_critical2 = self.ship2.getState(self.ship1.ship, self.asteroids, shuffle=SHUFFLE_AST)
        
        if done:
            game_done = 1
        else:
            game_done = 0 if self.ship1.alive and self.ship2.alive else 1
        
        # HUD Begin
        self.bullets_text.setText("bullets %d:%d [%d]"%(self.ship1.bullets_num, self.ship2.bullets_num, len(self.bullets)))
        self.life_text.setText("life %d:%d"%(self.ship1.life, self.ship2.life))
        self.text_agent1.setPos(self.ship1.ship.getPos().getX()/15-1, self.ship1.ship.getPos().getZ()/15+1)
        if SHOW_ARROW:
            self.ship_arrow.setPos(self.ship1.ship.getPos()/15)
            self.ship_arrow.setR(self.ship1.ship.getR()+np.rad2deg(ast_critical1.azimuth))
            self.ship_arrow1.setPos(self.ship2.ship.getPos()/15)
            self.ship_arrow1.setR(self.ship2.ship.getR()+np.rad2deg(ast_critical2.azimuth))
        # HUD End
        
        obs_state = (ship1_score, ship2_score, game_done,
                            len(ship1_state), *ship1_state, len(ship1_obs), *ship1_obs, 
                            len(ship2_state), *ship2_state, len(ship2_obs), *ship2_obs, 
                           )
        
        if self.p3d_proto:
            self.p3d_proto.set_game_state('{}f'.format(len(obs_state)), obs_state, game_done)
        
        return game_done
    
    def net_update_action(self):
        """ upd 获取的动作输入 """
        action = {}
        if self.p3d_proto:
            action.update( self.p3d_proto.get_game_action() )
        return action
    
    # Net Update End
    
    # Updates the positions of objects
    def updatePos(self, obj, dt):
        vel = self.getVelocity(obj)
        newPos = obj.getPos() + (vel * dt)

        # Check if the object is out of bounds. If so, wrap it
        radius = .5 * obj.getScale().getX()
        if newPos.getX() - radius > SCREEN_X:
            newPos.setX(-SCREEN_X)
        elif newPos.getX() + radius < -SCREEN_X:
            newPos.setX(SCREEN_X)
        if newPos.getZ() - radius > SCREEN_Y:
            newPos.setZ(-SCREEN_Y)
        elif newPos.getZ() + radius < -SCREEN_Y:
            newPos.setZ(SCREEN_Y)

        obj.setPos(newPos)

    # The handler when an asteroid is hit by a bullet
    def asteroidHit(self, index):
        # If the asteroid is small it is simply removed
        if self.asteroids[index].getScale().getX() <= AST_MIN_SCALE:
            self.asteroids[index].removeNode()
            # Remove the asteroid from the list of asteroids.
            del self.asteroids[index]
        else:
            # If it is big enough, divide it up into little asteroids.
            # First we update the current asteroid.
            asteroid = self.asteroids[index]
            newScale = asteroid.getScale().getX() * AST_SIZE_SCALE
            asteroid.setScale(newScale)  # Rescale it

            # The new direction is chosen as perpendicular to the old direction
            # This is determined using the cross product, which returns a
            # vector perpendicular to the two input vectors.  By crossing
            # velocity with a vector that goes into the screen, we get a vector
            # that is orthagonal to the original velocity in the screen plane.
            vel = self.getVelocity(asteroid)
            speed = vel.length() * AST_VEL_SCALE
            vel.normalize()
            vel = LVector3(0, 1, 0).cross(vel)
            vel *= speed
            self.setVelocity(asteroid, vel)

            # Now we create a new asteroid identical to the current one
            newAst = loadObject(scale=newScale)
            self.setVelocity(newAst, vel * -1)
            newAst.setPos(asteroid.getPos())
            newAst.setTexture(asteroid.getTexture(), 1)
            self.asteroids.append(newAst)


def on_exit():
    print("bye~")
    global reactor
    reactor.stop()


if __name__ == '__main__':
    # 3D rendering
    #~ demo = AsteroidsDemo()
    #~ demo.run()
    
    from twisted.internet.protocol import DatagramProtocol
    from twisted.internet.task import LoopingCall
    from twisted.internet import reactor
    
    from net_udp import Panda3dUDP
    
    #~ OBS_PIXEL = 1
    OBS_PIXEL = 0
    #~ ACT_DISC = 1
    #~ ACT_FMT = dict( fmt='I3f', name=['turn', 'accel', 'fire'] )
    ACT_FMT = dict( fmt='I6f', name=['turn', 'accel', 'fire', 'turn2', 'accel2', 'fire2'] )
    #~ ACT_FMT = dict( fmt='I2f', name=['turn', 'accel'] )
    
    demo = AsteroidsDemo(obs_pixel=OBS_PIXEL)
    taskMgr.step()
    
    p3d_proto = Panda3dUDP(p3d_step=taskMgr.step, action_fmt=ACT_FMT, obs_pixel=OBS_PIXEL)
    
    demo.p3d_proto = p3d_proto
    
    print("Port:", args.port)
    reactor.listenUDP(args.port, p3d_proto, maxPacketSize=8192)
    
    Desired_FPS = 30
    
    reactor.callWhenRunning(taskMgr.step)
    #~ LoopingCall(taskMgr.step).start(1 / Desired_FPS)
    reactor.run()

