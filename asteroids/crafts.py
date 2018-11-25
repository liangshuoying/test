from panda3d.core import TextNode, TransparencyAttrib
from panda3d.core import LPoint3, LVector3, LVector4, Vec3, Mat4
from direct.interval.MetaInterval import Sequence
from direct.interval.FunctionInterval import Wait, Func
from math import sin, cos, pi
import random
import numpy as np
import collections as col

from config import *

SHIP_LIFE = 3

CriticalAst = col.namedtuple('AstInfo', ['dist', 'azimuth'])

class Ship(object):
    
    def __init__(self, engine):
        self.eng = engine
        self.nextBullet = 0.0
        
        # Internal Game State
        self.bullets_fired = 0
        self.bullets_hit = 0
        self.bullets_num = BULLET_LIMIT
        self.ship_firing = 0
        #
        #~ self.life = 10
        self.life = SHIP_LIFE
        self.alive = 1
        self.game_score = 0
        self.crash = 0
        
        self.ship = self.loadObject("ship2.png", scale=2)
        self.setVelocity(self.ship, LVector3.zero())
    
    def getState(self, target, asteroids, invincible=0, shuffle=0, debug=0):
        #### Ship State Begin ###
        ship_m = Mat4()
        ship_m.invertFrom(self.ship.getMat())
        
        ship_z = self.ship.getMat().getRow3(2).normalized()
        ship_z = LVector3(ship_z)
        ship_pos = self.ship.getPos()
        ship_heading = self.ship.getR() * DEG_TO_RAD
        ship_vel = self.getVelocity(self.ship)
        ship_speed = ship_vel.length()
        
        # 本机指向相对正北方向的夹角 (-180, +180)
        ship_z_deg = ship_z.signedAngleDeg(LVector3(0, 0, 1), LVector3(0, -1, 0))
        ship_z_rad = ship_z_deg*DEG_TO_RAD
        
        # 本机速度矢量相对正北方向的夹角 (-180, +180)
        ship_vel_norm = ship_vel.normalized()
        ship_vel_deg = ship_vel_norm.signedAngleDeg(LVector3(0, 0, 1), LVector3(0, -1, 0))
        #~ ship_vel_deg = ship_vel_deg+360 if ship_vel_deg<0 else ship_vel_deg
        
        # 归一化的弹药余量指示
        ammo = float(self.bullets_num) / BULLET_LIMIT
        
        #~ ship_state = (ship_heading, ship_vel_deg*DEG_TO_RAD, ship_speed, ammo)  # test ammo
        #~ ship_state = (ship_heading, ship_vel_deg*DEG_TO_RAD, ship_speed, )  # best style
        ship_state = (ship_z_rad, ship_vel_deg*DEG_TO_RAD, ship_speed, )  # new style
        #~ ship_state = (ship_vel_deg*DEG_TO_RAD, ship_speed, ship_heading)
#         ship_state = (ship_speed, )
        ### Ship State End ###
        
        ### Target State Begin ###
        def get_ast(ast):
            ast_z = ast.getMat().getRow3(2).normalized()
            ast_z = LVector3(ast_z)
            ast_heading = ast.getR() * DEG_TO_RAD
            ast_pos = ast.getPos()
            
            # 目标与本机的相对位置矢量
            ast_azimuth = ast_pos - ship_pos
            ast_dist = ast_azimuth.length()
            ast_azimuth_norm = ast_azimuth.normalized()
            
            # 目标矢量和本机指向的夹角 (-180, +180)
            ast_azimuth_deg = ast_azimuth_norm.signedAngleDeg(ship_z, LVector3(0,-1,0))
            #~ ast_azimuth_deg = ast_azimuth_deg+360 if ast_azimuth_deg<0 else ast_azimuth_deg
            #~ ast_azimuth_deg = (self.ship.getR()+ast_azimuth_deg)%360
            ast_azimuth_rad = ast_azimuth_deg*DEG_TO_RAD
            
            # 目标矢量和目标机头指向的夹角 (-180, +180)
            ast_aim_deg = ast_azimuth_norm.signedAngleDeg(ast_z, LVector3(0,-1,0))
            ast_aim_rad = ast_aim_deg*DEG_TO_RAD
            
            # 目标机头指向和本机速度矢量的夹角 (-180, +180)
            ast_z_deg = ast_z.signedAngleDeg(ship_vel_norm, LVector3(0, -1, 0))
            ast_z_rad = ast_z_deg*DEG_TO_RAD
            
            # 目标速度矢量和本机指向的夹角 (-180, +180)
            ast_vel = self.getVelocity(ast)
            ast_vel_norm = ast_vel.normalized() 
            ast_vel_deg = ast_vel_norm.signedAngleDeg(ship_z, LVector3(0, -1, 0))
            #~ ast_vel_deg = ast_vel_norm.signedAngleDeg(LVector3(0, 0, 1), LVector3(0, -1, 0))
            #~ ast_vel_deg = ast_vel_deg+360 if ast_vel_deg<0 else ast_vel_deg
            ast_vel_rad = ast_vel_deg*DEG_TO_RAD
            
            # 目标速度矢量和本机速度矢量的夹角 (-180, +180)
            ast_vel_deg1 = ast_vel_norm.signedAngleDeg(ship_vel_norm, LVector3(0, -1, 0))
            ast_vel_rad1 = ast_vel_deg1*DEG_TO_RAD
            
            ast_speed = ast_vel.length()
            ast_vel_rel = (ast_vel - ship_vel)
            
#             ast_state = (ast_dist, ast_azimuth_rad, ast_aim_rad, ast_z_rad, ast_vel_rad, ast_vel_rad1, ast_speed)
            ast_state = (ast_dist, ast_azimuth_rad, ast_z_rad, ast_vel_rad, ast_vel_rad1)   # nice
            #~ ast_state = (ast_dist, ast_azimuth_rad, ast_vel_rad, )       # baseline
            
            ast_azimuth_lc = ship_m.xformVec(ast_azimuth).getXz()
            ast_vel_lc = ship_m.xformVec(ast_vel).getXz()
#             ast_vel_rel_lc = ship_m.xformVec(ast_vel_rel).getXz()
            
            #~ ast_state = (ast_dist, *ast_azimuth_lc, *ast_vel_lc, ast_azimuth_rad)
            
            return ast_state
        
        # Dogfight
        if DOG_FIGHT:
            ast_state = get_ast(target)
            ast_critical = CriticalAst(ast_state[0], ast_state[1])
            #~ ast_critical = ast_state
            #~ ast_state = ast_state [:]
        # Ast
        else:
            ast_state_list = []
            for ast in asteroids:
                ast_state = get_ast(ast)
                ast_state_list.append(ast_state)
                if not invincible:
                    self.checkCrash(ast)
            # sort by distance
            ast_state_list.sort( key=lambda ast:ast[0] )
            ast_critical = CriticalAst(ast_state_list[0][0], ast_state_list[0][1])
            #~ ast_critical = ast_state_list[0]
            # shuffle for test begin
#             ast_state_list = ast_state_list[:1]
            if shuffle:
                random.shuffle(ast_state_list)
            # test end
            # 压成一维数组
            ast_state_flat = []
            for s in ast_state_list:
                ast_state_flat.extend(s[:])
            ast_state = ast_state_flat
        
        ### Target State End ###
        
        # distance punish
        critical_dist = ast_critical[0]
        if self.ship_firing and critical_dist>GUN_RANGE and CONTROL_FIRE:
            self.game_score -= (1-GUN_RANGE/critical_dist)
            #~ game_score -= 0.1
        if critical_dist<AVOID_DIST and CONTROL_RAM:
            self.game_score -= (1-critical_dist/AVOID_DIST)
            #~ game_score -= 0.3    # good
        # punish end
        # DEBUG BEGIN
        if debug:
            ship_state = (1,2,3)
            ast_state = (4,5,6)
            self.game_score = 11
        # DEBUG END
        return ship_state, ast_state, self.game_score, ast_critical
    
    # This updates the ship's position. This is similar to the general update
    # but takes into account turn and thrust
    def updateShip(self, dt, a_turn=0, a_accel=-1, turn_ra=TURN_RATE):
        self.beginScore()
        
        heading = self.ship.getR()  # Heading is the roll value for this model
        
        if a_turn!=0:
            heading += dt * turn_ra * a_turn
            self.ship.setR(heading % 360)

        # Thrust causes acceleration in the direction the ship is currently
        # facing
        if a_accel>0:
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
        self.eng.updatePos(self.ship, dt)
    
    def updateFire(self, task, a_fire):
        # check to see if the ship can fire
        if a_fire>0 and task.time > self.nextBullet:
            self.fire(task.time)  # If so, call the fire function
            # And disable firing for a bit
            self.nextBullet = task.time + BULLET_REPEAT
            # Bullet Limit Begin
            self.ship_firing = 1
            self.bullets_fired += 1
            self.bullets_num = BULLET_LIMIT-self.bullets_fired
            # Bullet Limit End
        else:
            self.ship_firing = 0
    
    # Creates a bullet and adds it to the bullet list
    def fire(self, time):
        direction = DEG_TO_RAD * self.ship.getR()
        pos = self.ship.getPos()
        bullet = self.loadObject("bullet.png", scale=.4)  # Create the object
        bullet.setPos(pos)
        # Velocity is in relation to the ship
        vel = (self.getVelocity(self.ship) +
               (LVector3(sin(direction), 0, cos(direction)) *
                BULLET_SPEED))
        self.setVelocity(bullet, vel)
        # Set the bullet expiration time to be a certain amount past the
        # current time
        self.setExpires(bullet, time + BULLET_LIFE)
        self.setAttacker(bullet, id(self))
        # Finally, add the new bullet to the list
        self.eng.bullets.append(bullet)
    
    def beginScore(self):
        """ 注意,每步开始都要归零 """
        self.game_score = 0
    
    def checkHit(self, bullet):
        if self.getAttacker(bullet)==id(self):
            return 0
        if ((bullet.getPos() - self.ship.getPos()).lengthSquared() <
            (((bullet.getScale().getX() + self.ship.getScale().getX())
              * .5) ** 2)):
            # Schedule the bullet for removal
            self.setExpires(bullet, 0)
            # check life
            if self.life>0:
                self.life -= 1
                self.game_score -= 10
            if self.life<=0:
                self.alive = 0
                self.game_score -= 100
            # life end
            return 1
        return 0
    
    def checkCrash(self, ast):
        shipSize = self.ship.getScale().getX()
        if ((self.ship.getPos() - ast.getPos()).lengthSquared() <
                (((shipSize + ast.getScale().getX()) * .5) ** 2)):
            # Ship is no longer alive
            self.alive = 0
            self.life = 0
            self.game_score -= 100
            # end
            self.crash += 1
            return 1
        return 0
    
    def reset(self, restart=0):
        # bullet limit reset
        self.bullets_fired = 0
        self.bullets_hit = 0
        self.bullets_num = BULLET_LIMIT
        #~ self.bullets_text.setText("bullets %d"%self.bullets_num)
        self.ship_firing = 0
        # bullet limit end
        self.life = SHIP_LIFE
        self.alive = 1
        self.game_score = 0
        if restart:
            self.crash = 0
        
        self.ship.hide()
        v = np.random.uniform(-MAX_VEL*0.6, MAX_VEL*0.6)
        self.setVelocity(self.ship, LVector3(v, 0, v))
        self.ship.setR( random.randrange(0,360) )
        self.ship.setX( np.random.uniform(-SCREEN_X,SCREEN_X) )
        self.ship.setZ( np.random.uniform(-SCREEN_X,SCREEN_X) )
        self.ship.show()
        #~ Sequence(Func(self.ship.setR, 0),  # Reset heading
                        #~ Func(self.ship.setX, 0),  # Reset position X
                        #~ Func(self.ship.setZ, 0),
                        #~ Func(self.ship.show),     # Show the ship
                    #~ ).start()
    
    def show(self):
        self.ship.show()
    
    def hide(self):
        self.ship.hide()
    
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
    
    def loadObject(self, tex=None, pos=LPoint3(0, 0), depth=SPRITE_POS, scale=1,
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


