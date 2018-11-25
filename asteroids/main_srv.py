#!/usr/bin/env python

# Author: Shao Zhang, Phil Saltzman, and Greg Lindley
# Last Updated: 2015-03-13
#
# This tutorial demonstrates the use of tasks. A task is a function that
# gets called once every frame. They are good for things that need to be
# updated very often. In the case of asteroids, we use tasks to update
# the positions of all the objects, and to check if the bullets or the
# ship have hit the asteroids.
#
# Note: This definitely a complicated example. Tasks are the cores of
# most games so it seemed appropriate to show what a full game in Panda
# could look like.

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

import argparse

parser = argparse.ArgumentParser(description='Panda3d')
parser.add_argument('--port', type=int, default=8880, help='Server Port')
parser.add_argument('--render', type=int, default=0, help='Render Me')
args = parser.parse_args()

#~ WIN_W, WIN_H = 640, 480
WIN_W, WIN_H = 512, 512
if not args.render:
    WIN_W, WIN_H = 400, 400
    loadPrcFileData("", "window-type offscreen")
#~ loadPrcFileData("", "load-display p3tinydisplay")
#~ loadPrcFileData("", "framebuffer-hardware 0")
#~ loadPrcFileData("", "framebuffer-software 1")
loadPrcFileData('', 'win-size %d %d'%(WIN_W, WIN_H))
#~ loadPrcFileData('', 'show-frame-rate-meter 1')

# Constants that will control the behavior of the game. It is good to
# group constants like this so that they can be changed once without
# having to find everywhere they are used in code
SPRITE_POS = 55     # At default field of view and a depth of 55, the screen
# dimensions is 40x30 units
#~ SCREEN_X = 20       # Screen goes from -20 to 20 on X
SCREEN_X = 15       # Screen goes from -20 to 20 on X
SCREEN_Y = 15       # Screen goes from -15 to 15 on Y
TURN_RATE = 360     # Degrees ship can turn in 1 second
ACCELERATION = 10   # Ship acceleration in units/sec/sec
MAX_VEL = 6         # Maximum ship velocity in units/sec
MAX_VEL_SQ = MAX_VEL ** 2  # Square of the ship velocity
DEG_TO_RAD = pi / 180  # translates degrees to radians for sin and cos
#~ BULLET_LIFE = 2     # How long bullets stay on screen before removed
BULLET_LIFE = 0.7     # How long bullets stay on screen before removed
BULLET_REPEAT = .2  # How often bullets can be fired
BULLET_SPEED = 10   # Speed bullets move
AST_INIT_VEL = 1    # Velocity of the largest asteroids
AST_INIT_SCALE = 3  # Initial asteroid scale
AST_VEL_SCALE = 2.2  # How much asteroid speed multiplies when broken up
AST_SIZE_SCALE = .6  # How much asteroid scale changes when broken up
AST_MIN_SCALE = 1.1  # If and asteroid is smaller than this and is hit,
# it disapears instead of splitting up
#~ AST_NUM = 10
#~ AST_NUM = 1
AST_NUM = 2
#~ AST_NUM = 3
BULLET_LIMIT = 100
GUN_RANGE = 7.
#~ AVOID_DIST = 6.0
#~ AVOID_DIST = 5.0
AVOID_DIST = 4.0
#~ AVOID_DIST = 3.0


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
        
        # Internal Game State
        self.bullets_fired = 0
        self.bullets_hit = 0
        self.bullets_num = BULLET_LIMIT
        self.ship_firing = 0
        #
        self.ship_arrow = P3DCreateAxes(length=GUN_RANGE/SCREEN_X)
        self.ship_arrow1 = P3DCreateAxes(length=AVOID_DIST/SCREEN_X)
        
        #~ self.ast_arrow = P3DCreateAxes(length=GUN_RANGE/SCREEN_X)
        #~ self.ast_arrow1 = P3DCreateAxes(length=AVOID_DIST/SCREEN_X)
        
        self.bullets_text = OnscreenText(text="bullets %d"%self.bullets_num,
                                  parent=base.a2dBottomRight, scale=.1,
                                  align=TextNode.ALeft, pos=(-1.99, 1.9),
                                  fg=(0, 1, 0, 1), shadow=(0, 0, 0, 0.5))
        
        # This code puts the standard title and instruction text on screen
        #~ self.title = OnscreenText(text="Panda3D: Tutorial - Tasks",
                                  #~ parent=base.a2dBottomRight, scale=.07,
                                  #~ align=TextNode.ARight, pos=(-0.1, 0.1),
                                  #~ fg=(1, 1, 1, 1), shadow=(0, 0, 0, 0.5))
        #~ self.escapeText = genLabelText("ESC: Quit", 0)
        #~ self.leftkeyText = genLabelText("[Left Arrow]: Turn Left (CCW)", 1)
        #~ self.rightkeyText = genLabelText("[Right Arrow]: Turn Right (CW)", 2)
        #~ self.upkeyText = genLabelText("[Up Arrow]: Accelerate", 3)
        #~ self.spacekeyText = genLabelText("[Space Bar]: Fire", 4)

        # Disable default mouse-based camera control.  This is a method on the
        # ShowBase class from which we inherit.
        self.disableMouse()

        # Load the background starfield.
        self.setBackgroundColor((0, 0, 0, 1))
        self.bg = loadObject("stars.jpg", scale=146, depth=200,
                             transparency=False)

        # Load the ship and set its initial velocity.
        self.ship = loadObject("ship.png", scale=2)
        self.setVelocity(self.ship, LVector3.zero())

        # A dictionary of what keys are currently being pressed
        # The key events update this list, and our task will query it as input
        self.keys = {"turnLeft": 0, "turnRight": 0,
                     "accel": 0, "fire": 0}

        #~ self.accept("escape", sys.exit)  # Escape quits
        self.accept("escape", on_exit)  # Escape quits
        # Other keys events set the appropriate value in our key dictionary
        self.accept("arrow_left",     self.setKey, ["turnLeft", 1])
        self.accept("arrow_left-up",  self.setKey, ["turnLeft", 0])
        self.accept("arrow_right",    self.setKey, ["turnRight", 1])
        self.accept("arrow_right-up", self.setKey, ["turnRight", 0])
        self.accept("arrow_up",       self.setKey, ["accel", 1])
        self.accept("arrow_up-up",    self.setKey, ["accel", 0])
        self.accept("space",          self.setKey, ["fire", 1])

        # Now we create the task. taskMgr is the task manager that actually
        # calls the function each frame. The add method creates a new task.
        # The first argument is the function to be called, and the second
        # argument is the name for the task.  It returns a task object which
        # is passed to the function each frame.
        self.gameTask = taskMgr.add(self.gameLoop, "gameLoop")

        # Stores the time at which the next bullet may be fired.
        self.nextBullet = 0.0

        # This list will stored fired bullets.
        self.bullets = []

        # Complete initialization by spawning the asteroids.
        self.spawnAsteroids()

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

        # If the ship is not alive, do nothing.  Tasks return Task.cont to
        # signify that the task should continue running. If Task.done were
        # returned instead, the task would be removed and would no longer be
        # called every frame.
        if not self.alive:
            return Task.cont
        
        # Net Update Action
        action = self.net_update_action()
        a_turn = action.get('turn', 0)
        a_accel = action.get('accel', -1)
        a_fire = action.get('fire', -1)
        game_score = 0.
        # Net Update End
        
        # update ship position
        self.updateShip(dt, a_turn, a_accel)

        # check to see if the ship can fire
        if (self.keys["fire"] or a_fire>0) and task.time > self.nextBullet:
            self.fire(task.time)  # If so, call the fire function
            # And disable firing for a bit
            self.nextBullet = task.time + BULLET_REPEAT
            # Bullet Limit Begin
            self.ship_firing = 1
            self.bullets_fired += 1
            self.bullets_num = BULLET_LIMIT-self.bullets_fired
            self.bullets_text.setText("bullets %d"%self.bullets_num)
            if self.bullets_fired > BULLET_LIMIT and 0:
                game_score = -50
                if self.bullets_hit:
                    hit_rate = 0.9
                    game_score = (float(self.bullets_hit)/self.bullets_fired-hit_rate) * 10
                    if game_score<0:
                        game_score *= 10
                self.net_update_state(game_score, done=1)
                self.resetGame(wait=0)
                return Task.cont 
            # Bullet Limit End
        else:
            self.ship_firing = 0
        # Remove the fire flag until the next spacebar press
        self.keys["fire"] = 0

        # update asteroids
        for obj in self.asteroids:
            self.updatePos(obj, dt)

        # update bullets
        newBulletArray = []
        for obj in self.bullets:
            self.updatePos(obj, dt)  # Update the bullet
            # Bullets have an experation time (see definition of fire)
            # If a bullet has not expired, add it to the new bullet list so
            # that it will continue to exist.
            if self.getExpires(obj) > task.time:
                newBulletArray.append(obj)
            else:
                obj.removeNode()  # Otherwise, remove it from the scene.
                # Punish Begin
                #~ game_score -= 1
                # Punish End
        # Set the bullet array to be the newly updated array
        self.bullets = newBulletArray

        # Check bullet collision with asteroids
        # In short, it checks every bullet against every asteroid. This is
        # quite slow.  A big optimization would be to sort the objects left to
        # right and check only if they overlap.  Framerate can go way down if
        # there are many bullets on screen, but for the most part it's okay.
        for bullet in self.bullets:
            # This range statement makes it step though the asteroid list
            # backwards.  This is because if an asteroid is removed, the
            # elements after it will change position in the list.  If you go
            # backwards, the length stays constant.
            for i in range(len(self.asteroids) - 1, -1, -1):
                asteroid = self.asteroids[i]
                # Panda's collision detection is more complicated than we need
                # here.  This is the basic sphere collision check. If the
                # distance between the object centers is less than sum of the
                # radii of the two objects, then we have a collision. We use
                # lengthSquared() since it is faster than length().
                if ((bullet.getPos() - asteroid.getPos()).lengthSquared() <
                    (((bullet.getScale().getX() + asteroid.getScale().getX())
                      * .5) ** 2)):
                    # Schedule the bullet for removal
                    self.setExpires(bullet, 0)
                    self.asteroidHit(i)      # Handle the hit
                    # Net Update Score
                    #~ game_score += 1.
                    game_score += 10.
                    self.bullets_hit += 1
                    # Net Update End

        # Now we do the same collision pass for the ship
        shipSize = self.ship.getScale().getX()
        for ast in self.asteroids:
            # Same sphere collision check for the ship vs. the asteroid
            if ((self.ship.getPos() - ast.getPos()).lengthSquared() <
                    (((shipSize + ast.getScale().getX()) * .5) ** 2)):
                # If there is a hit, clear the screen and schedule a restart
                self.alive = False         # Ship is no longer alive
                # Remove every object in asteroids and bullets from the scene
                
                # Net Update State Before Reset
                game_score -= 100
                self.net_update_state(game_score, done=1)
                
                self.resetGame()
                return Task.cont 
                # Net Update End
                
                for i in self.asteroids + self.bullets:
                    i.removeNode()
                self.bullets = []          # Clear the bullet list
                self.ship.hide()           # Hide the ship
                # Reset the velocity
                self.setVelocity(self.ship, LVector3(0, 0, 0))
                Sequence(Wait(0),          # Wait 2 seconds
                         Func(self.ship.setR, 0),  # Reset heading
                         Func(self.ship.setX, 0),  # Reset position X
                         # Reset position Y (Z for Panda)
                         Func(self.ship.setZ, 0),
                         Func(self.ship.show),     # Show the ship
                         Func(self.spawnAsteroids)).start()  # Remake asteroids
                return Task.cont

        # If the player has successfully destroyed all asteroids, respawn them
        if len(self.asteroids) == 0:
            self.spawnAsteroids()
        
        # Net Update State
        self.net_update_state(game_score)
        # Net Update End
        
        return Task.cont    # Since every return is Task.cont, the task will
        # continue indefinitely
    
    # Net Update Begin
    
    def resetGame(self, wait=0):
        # bullet limit reset
        self.bullets_fired = 0
        self.bullets_hit = 0
        self.bullets_num = BULLET_LIMIT
        self.bullets_text.setText("bullets %d"%self.bullets_num)
        self.ship_firing = 0
        # bullet limit end
        self.alive = False
        for i in self.asteroids + self.bullets:
            i.removeNode()
        self.bullets = []          # Clear the bullet list
        self.ship.hide()           # Hide the ship
        # Reset the velocity
        self.setVelocity(self.ship, LVector3(0, 0, 0))
        Sequence(Wait(wait),          # Wait 2 seconds
                 Func(self.ship.setR, 0),  # Reset heading
                 Func(self.ship.setX, 0),  # Reset position X
                 # Reset position Y (Z for Panda)
                 Func(self.ship.setZ, 0),
                 Func(self.ship.show),     # Show the ship
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
    
    def net_update_state(self, game_score, done=0):
        """ 游戏状态, 输出到udp """
        ship_z = self.ship.getMat().getRow3(2).normalized() 
        ship_pos = self.ship.getPos()
        ship_heading = self.ship.getR() * DEG_TO_RAD
        ship_vel = self.getVelocity(self.ship)
        ship_speed = ship_vel.length()
        
        ship_vel_norm = ship_vel.normalized() 
        ship_vel_deg = ship_vel_norm.signedAngleDeg(LVector3(0, 0, 1), LVector3(0, -1, 0))
        ship_vel_deg = ship_vel_deg+360 if ship_vel_deg<0 else ship_vel_deg
        
        ammo = float(self.bullets_num) / BULLET_LIMIT
        
        #~ ship_state = (ship_heading, ship_vel_deg*DEG_TO_RAD, ship_speed, ammo)  # test ammo
        ship_state = (ship_heading, ship_vel_deg*DEG_TO_RAD, ship_speed, )  # best style
        #~ ship_state = (ship_vel_deg*DEG_TO_RAD, ship_speed, ship_heading)
        
        if done:
            game_done = 1
        else:
            game_done = 0 if self.alive else 1
        
        ast_state = []
        for ast in self.asteroids:
            ast_heading = ast.getR() * DEG_TO_RAD
            ast_pos = ast.getPos()
            ast_azimuth = ast_pos - ship_pos
            ast_dist = ast_azimuth.length()
            ast_azimuth_norm = ast_azimuth.normalized()
            ship_ref = ship_z
            #~ ship_ref = ship_z if ship_vel.lengthSquared()==0 else ship_vel.normalized()
            ast_azimuth_deg = ast_azimuth_norm.signedAngleDeg(ship_ref, LVector3(0,-1,0))
            #~ ast_azimuth_deg = ast_azimuth_deg+360 if ast_azimuth_deg<0 else ast_azimuth_deg
            #~ ast_azimuth_deg = (self.ship.getR()+ast_azimuth_deg)%360
            ast_azimuth_rad = ast_azimuth_deg*DEG_TO_RAD
            
            ast_vel = self.getVelocity(ast)
            ast_vel_norm = ast_vel.normalized() 
            ast_vel_deg = ast_vel_norm.signedAngleDeg(LVector3(0, 0, 1), LVector3(0, -1, 0))
            #~ ast_vel_deg = ast_vel_deg+360 if ast_vel_deg<0 else ast_vel_deg
            ast_vel_rad = ast_vel_deg*DEG_TO_RAD
            
            ast_vel_rel = (ast_vel - ship_vel)
            # 5 floats
            ast_state.append( (ast_dist, ast_azimuth_rad, ast_vel_rad, ast_heading) )    # best style
            
        # sort by distance
        ast_state.sort( key=lambda ast:ast[0] )
        # 只取5个最近的小行星
        
        # HUD Begin
        self.ship_arrow.setPos(self.ship.getPos()/15)
        self.ship_arrow.setR(self.ship.getR())
        self.ship_arrow1.setPos(self.ship.getPos()/15)
        self.ship_arrow1.setR(ship_vel_deg)
        
        #~ self.ast_arrow.setPos(self.ast.getPos()/15)
        #~ self.ast_arrow.setR(self.ast.getR())
        #~ self.ast1_arrow1.setPos(self.ast.getPos()/15)
        #~ self.ast1_arrow1.setR(ast_vel_deg)
        #~ self.ss_vel.setPos(self.ship.getPos()/15)
        #~ self.ss.setR(self.ship.getR())
        #~ self.ss_vel.setR(ship_vel_deg)
        # HUD End
        
        # distance punish
        critical_dist = ast_state[0][0]
        if self.ship_firing and critical_dist>GUN_RANGE and 0:
            game_score -= (1-GUN_RANGE/critical_dist)
            #~ game_score -= 0.1
        if critical_dist<AVOID_DIST and 0:
            game_score -= (1-critical_dist/AVOID_DIST)**2
            #~ game_score -= 0.3    # good
        # punish end
        
        # Target tracking begin
        #~ ast0 = ast_state[0]
        #~ if self.alive:
            #~ reward = 1. - ast0[-1]
            #~ if reward<0:
                #~ reward /= SCREEN_X*2
        #~ else:
            #~ reward = 100
        # test end
        
        # use CNN
        if self.obs_pixel:
            # states begin
            state = (game_score, game_done, ship_heading)
            obs = self.getFrame()
            # states end
            if self.p3d_proto:
                self.p3d_proto.set_game_state_pixel('7056s3f', obs.tobytes(), state, game_done)
            return
        # END and return
        
        ast_state = ast_state[:1]
        #~ ast_state = ast_state[:5]
        # 如果没5个就插入补齐
        #~ while len(ast_state)<5:
            #~ ast_state.append( (0, 0, -1) )
        
        # 压成一维数组
        ast_state_flat = []
        for s in ast_state:
            ast_state_flat.extend(s[:])
        
        #~ obs_state = (*ast_state_flat, *ship_state, reward, game_done)   # best style
        obs_state = (*ship_state, *ast_state_flat, game_score, game_done)
        
        if self.p3d_proto:
            #~ self.p3d_proto.set_game_state('8f', obs, state, game_done)
            self.p3d_proto.set_game_state('{}f'.format(len(obs_state)), obs_state, game_done)
    
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

    # This updates the ship's position. This is similar to the general update
    # but takes into account turn and thrust
    def updateShip(self, dt, a_turn=0, a_accel=-1):
        heading = self.ship.getR()  # Heading is the roll value for this model
        
        # Change heading if left or right is being pressed
        if self.keys["turnRight"]:
            heading += dt * TURN_RATE
            self.ship.setR(heading % 360)
        elif self.keys["turnLeft"]:
            heading -= dt * TURN_RATE
            self.ship.setR(heading % 360)
        elif a_turn!=0:
            heading += dt * TURN_RATE * a_turn
            self.ship.setR(heading % 360)

        # Thrust causes acceleration in the direction the ship is currently
        # facing
        if self.keys["accel"] or (a_accel>0):
            heading_rad = DEG_TO_RAD * heading
            # This builds a new velocity vector and adds it to the current one
            # relative to the camera, the screen in Panda is the XZ plane.
            # Therefore all of our Y values in our velocities are 0 to signify
            # no change in that direction.
            newVel = \
                LVector3(sin(heading_rad), 0, cos(heading_rad)) * \
                ACCELERATION * dt * a_accel
            newVel += self.getVelocity(self.ship)
            # Clamps the new velocity to the maximum speed. lengthSquared() is
            # used again since it is faster than length()
            if newVel.lengthSquared() > MAX_VEL_SQ:
                newVel.normalize()
                newVel *= MAX_VEL
            self.setVelocity(self.ship, newVel)

        # Finally, update the position as with any other object
        self.updatePos(self.ship, dt)

    # Creates a bullet and adds it to the bullet list
    def fire(self, time):
        direction = DEG_TO_RAD * self.ship.getR()
        pos = self.ship.getPos()
        bullet = loadObject("bullet.png", scale=.4)  # Create the object
        bullet.setPos(pos)
        # Velocity is in relation to the ship
        vel = (self.getVelocity(self.ship) +
               (LVector3(sin(direction), 0, cos(direction)) *
                BULLET_SPEED))
        self.setVelocity(bullet, vel)
        # Set the bullet expiration time to be a certain amount past the
        # current time
        self.setExpires(bullet, time + BULLET_LIFE)

        # Finally, add the new bullet to the list
        self.bullets.append(bullet)

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
    ACT_FMT = dict( fmt='I3f', name=['turn', 'accel', 'fire'] )
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

