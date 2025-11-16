import sys , os, time
from math import *
from mathutils import *
import numpy as np
from numpy_da import DynamicArray
from random import random

import bpy

DEBUG = True
start_time = time.time()
q_zero = Quaternion((1.0,0.0,0.0,0.0))
q_zero0 = Quaternion((-1.0,0.0,0.0,0.0))
M0 = Matrix([[1.0,0.0,0.0,0.0]
                ,[0.0,1.0,0.0,0.0]
                ,[0.0,0.0,1.0,0.0]
                ,[0.0,0.0,0.0,1.0]])
def _normalize_angle(angle):
    """Нормалізує кут в діапазон [-pi, pi]."""
    return atan2(sin(angle), cos(angle))

def rotation_matrix_from_euler_angles(roll, pitch, yaw):
    """
    Створення матриці обертання з кутів Ейлера (roll, pitch, yaw)
    """
    R_x = np.array([[1, 0, 0],
                    [0, cos(roll), -sin(roll)],
                    [0, sin(roll), cos(roll)]])

    R_y = np.array([[cos(pitch), 0, sin(pitch)],
                    [0, 1, 0],
                    [-sin(pitch), 0, cos(pitch)]])

    R_z = np.array([[cos(yaw), -sin(yaw), 0],
                    [sin(yaw), cos(yaw), 0],
                    [0, 0, 1]])

    R = np.dot(R_z, np.dot(R_y, R_x))
    return R

def euler_angles_from_rotation_matrix(R):
    """
    Отримання кутів Ейлера (roll, pitch, yaw) з матриці обертання
    """
    sy = sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        roll = atan2(R[2, 1], R[2, 2])
        pitch = atan2(-R[2, 0], sy)
        yaw = atan2(R[1, 0], R[0, 0])
    else:
        roll = atan2(-R[1, 2], R[1, 1])
        pitch = atan2(-R[2, 0], sy)
        yaw = 0

    return roll, pitch, yaw

def relative_to_global_euler(relative_euler, object_euler):
    """
    Перетворення відносних кутів Ейлера в глобальні
    """
    Rx_object = rotation_matrix_from_euler_angles(object_euler[0], object_euler[1], object_euler[2])
    Rx_relative = rotation_matrix_from_euler_angles(relative_euler[0], relative_euler[1], relative_euler[2])
    Rx_global = np.dot(Rx_object, Rx_relative)
    return euler_angles_from_rotation_matrix(Rx_global)

def global_to_relative_euler(global_euler, object_euler):
    """
    Перетворення глобальних кутів Ейлера в відносні
    """
    Rx_object = rotation_matrix_from_euler_angles(object_euler[0], object_euler[1], object_euler[2])
    Rx_global = rotation_matrix_from_euler_angles(global_euler[0], global_euler[1], global_euler[2])
    Rx_relative = np.dot(np.transpose(Rx_object), Rx_global)
    return euler_angles_from_rotation_matrix(Rx_relative)


def euler_from_quaternion(q,radians = True):
            w = q[0]
            x = q[1]
            y = q[2]
            z = q[3]
            """
            Convert a quaternion into euler angles (roll, pitch, yaw)
            roll is rotation around x in radians (counterclockwise)
            pitch is rotation around y in radians (counterclockwise)
            yaw is rotation around z in radians (counterclockwise)
            """
            t0 = +2.0 * (w * x + y * z)
            t1 = +1.0 - 2.0 * (x * x + y * y)
            roll_x = atan2(t0, t1)

            t2 = +2.0 * (w * y - z * x)
            t2 = +1.0 if t2 > +1.0 else t2
            t2 = -1.0 if t2 < -1.0 else t2
            pitch_y = asin(t2)

            t3 = +2.0 * (w * z + x * y)
            t4 = +1.0 - 2.0 * (y * y + z * z)
            yaw_z = atan2(t3, t4)
            if radians:
                return [roll_x,pitch_y, yaw_z] # in radians
            else:
                return [degrees(roll_x), degrees(pitch_y), degrees(yaw_z)] # in degrees


class Force :
    def __init__(self, vector = [0.0,0.0,1.0], strength = 1.0, offset= [0.0,0.0,0.0], local = True):
        global DEBUG
        self.ID = start_time - time.time()
        if DEBUG:
            print ('Force object',self.ID)
        self.strength = strength
        self.offset = Vector(offset)
        self.vector = Vector(vector)
        self.local = local
    @property
    def get_strength(self):
        return self.strength
    @property
    def get_offset(self):
        return self.offset
        #return np.array([self.matrix_base[0][3],self.matrix_base[1][3],self.matrix_base[2][3]])
    #@property
    #def rotation_euler(self):
        #return Rotation.from_matrix([self.matrix_base[0][:3],self.matrix_base[1][:3],self.matrix_base[2][:3]]).as_euler('xyz', degrees=False)
    @property
    def get_vector (self):
        return self.vector
    @property
    def get_force_vector (self):
        return self.vector*self.strength
    def set_offset(self, pos= [0.0,0.0,0.0]):
        self.offset = Vector(pos)
    def set_vector(self, pos= [0.0,0.0,0.0]):
        self.vector = Vector(pos)
    def set_force_vector(self, pos= [0.0,0.0,0.0]):
        scalar = (pos[0]**2+ pos[1]**2+pos[2]**2)**0.5
        if scalar >0.001:
            self.vector = Vector(pos)/scalar
            self.strength = scalar
    def add_force_vector(self, pos= [0.0,0.0,0.0]):
        vec = self.vector*self.strength
        vec[0] += pos[0]
        vec[1] += pos[1]
        vec[2] += pos[2]
        scalar = (vec[0]**2+ vec[1]**2+vec[2]**2)**0.5
        if scalar >0.001:
            self.vector = Vector(vec)/scalar
            self.strength = scalar
        else:
            self.vector = Vector([0.0,0.0,1.0])
            self.strength = 0.0
    def set_strength(self,strength = 0.0):
        self.strength = strength
    def apply_rotation(self,rot,order = 'xyz'):
        if rot != None:
            possible = False
            if len (rot) ==3:
                r = Vector(order)
                possible = True
            elif len (rot) ==4:
                r = Quaternion(rot)#.as_matrix()
                possible = True
            if possible:
                self.vector = self.vector @ r
                self.offset = self.offset @ r
class Acceleration (Force):
    def __init__(self, vector = [0.0,0.0,1.0], strength = 1.0, offset= [0.0,0.0,0.0], local = True, types = 'NONE'):
        super().__init__(vector=vector,strength=strength,offset=offset,local=local)
        if types in ['NONE','GRAVITY','DEFORM','MAGIC']:
            self.types = types
        else:
            self.types = 'NONE'
    def set_type(self, types = 'NONE'):
        if types in ['NONE','GRAVITY','DEFORM','MAGIC']:
            self.types = types
        else:
            self.types = 'NONE'
class Joint :
    def __init__(self, Body_1,Body_2=None, strength = 1.0, stiffness = 1.0,torque = 0.5,friction = 0.5, pilot_1= [0.0,0.0,0.0], pilot_2 =[0.0,0.0,0.0], local = True, flex = True, rigid = True,
                collision =False, max_limit_rot = Euler([0.0,0.0,0.0]),min_limit_rot = Euler([0.0,0.0,0.0]),min_length = 0.0,max_length = 0.0,active = True):
        global DEBUG
        self.ID = start_time - time.time()
        if DEBUG:
            print ('Joint object',self.ID)
        self.strength = strength
        self.stiffness = stiffness
        self.torque = torque
        self.friction = friction
        self.Body_1 = Body_1
        self.Body_2 = Body_2
        self.pilot_1 = Vector(pilot_1)
        self.pilot_2 = Vector(pilot_2)
        self.max_limit_rot = Euler([0.0,0.0,0.0])
        self.min_limit_rot = Euler([0.0,0.0,0.0])
        self.local = local
        self.flex = flex
        self.rigid = rigid
        self.collision = collision
        error = False
        if len (max_limit_rot) == 3 and len (min_limit_rot) == 3:
            for i in range (len(self.max_limit_rot)):
                if max_limit_rot[i] < min_limit_rot[i]:
                    error = True
                    break
        else:
            error = True
        if error:
            self.max_limit_rot = Euler([0.0,0.0,0.0])
            self.min_limit_rot = Euler([0.0,0.0,0.0])
            print ('Coution! Uncorrect angle limits!')
        else:
            self.max_limit_rot = max_limit_rot
            self.min_limit_rot = min_limit_rot
        self.min_length = min_length
        self.max_length = max_length
        self.active = active
    def get_distance (self, body = None):
        if body == self.Body_1:
            if self.Body_1 != None:
                offset_1 = self.pilot_1 @ self.Body_1.rotation_matrix.transposed()
                pos_1 = self.Body_1.position
            else:
                offset_1 = self.pilot_1
                pos_1 = Vector([0.0,0.0,0.0])
            if self.Body_2 != None:
                offset_2= self.pilot_2 @ self.Body_2.rotation_matrix.transposed()
                pos_2 = self.Body_2.position
            else:
                offset_2 = self.pilot_2
                pos_2 = Vector([0.0,0.0,0.0])
            vec = pos_2 - pos_1 + offset_2 - offset_1
            if self.rigid:
                return vec
            else:
                if vec.length < self.min_length:
                    if vec.length > self.max_length:
                        return vec
                    else:
                        self.active = False
                        return Vector([0.0,0.0,0.0])
                else:
                    return Vector([0.0,0.0,0.0])
                
        elif body == self.Body_2:
            if self.Body_1 != None:
                offset_1 = self.pilot_1# @ self.Body_1.rotation_matrix#.inverted()
                pos_1 = self.Body_1.position #+ offset_1
            else:
                offset_1 = self.pilot_1
                pos_1 = Vector([0.0,0.0,0.0])
            if self.Body_2 != None:
                offset_2= self.pilot_2# @ self.Body_2.rotation_matrix#.inverted()
                pos_2 = self.Body_2.position #+ offset_2
            else:
                offset_2 = self.pilot_2
                pos_2 = Vector([0.0,0.0,0.0])
            vec = pos_1 - pos_2 + offset_1 - offset_2
            if self.rigid:
                return vec
            else:
                if vec.length < self.min_length:
                    return vec
                else:
                    if vec.length > self.max_length:
                        self.active = False
                        return Vector([0.0,0.0,0.0])
                    else:
                        return vec
                    
    def get_rotation_difference (self, body = None):
        '''def normalize(v):
            norm=np.linalg.norm(v)
            if norm==0:
                norm=np.finfo(v.dtype).eps
            return v/norm'''
        def q_to_axisangle(q):
            w, v = q[0], q.vector
            theta = acos(w) * 2.0
            return normalize(v), theta
        def euler_from_quaternion(q,radians = True):
            w = q[0]
            x = q[1]
            y = q[2]
            z = q[3]
            """
            Convert a quaternion into euler angles (roll, pitch, yaw)
            roll is rotation around x in radians (counterclockwise)
            pitch is rotation around y in radians (counterclockwise)
            yaw is rotation around z in radians (counterclockwise)
            """
            t0 = +2.0 * (w * x + y * z)
            t1 = +1.0 - 2.0 * (x * x + y * y)
            roll_x = atan2(t0, t1)

            t2 = +2.0 * (w * y - z * x)
            t2 = +1.0 if t2 > +1.0 else t2
            t2 = -1.0 if t2 < -1.0 else t2
            pitch_y = asin(t2)

            t3 = +2.0 * (w * z + x * y)
            t4 = +1.0 - 2.0 * (y * y + z * z)
            yaw_z = atan2(t3, t4)
            if radians:
                return Vector([roll_x,pitch_y, yaw_z]) # in radians
            else:
                return Vector([degrees(roll_x), degrees(pitch_y), degrees(yaw_z)]) # in degrees
        x = Vector([1.0,0.0,0.0])
        y = Vector ([0.0,1.0,0.0])
        z = Vector ([0.0,0.0,1.0])
        if body == self.Body_1:
            if self.Body_1 != None:
                rot_1 = self.Body_1.rotation_matrix
            else:
                rot_1 = M0
            if self.Body_2 != None:
                rot_2 = self.Body_2.rotation_matrix
            else:
                rot_2 = M0
            if rot_2 != rot_1:
                rot = rot_1.inverted() @ rot_2
                res = rot.to_euler('XYZ')
                #res = Euler([0.0,0.0,0.0])
            else:
                res = Euler([0.0,0.0,0.0])
            if self.flex:
                print ('flex', res)
                for i in range (len(res)):
                    if self.max_limit_rot[i] >= res[i] >= self.min_limit_rot[i]  :
                        res[i] = 0.0
                        print ('flex_1_1')
                    elif res[i] < self.min_limit_rot[i]:
                        res[i] += self.min_limit_rot[i]
                        print ('flex_1_2_minimum')
                    elif res[i] > self.max_limit_rot[i]:
                        res[i] -= self.max_limit_rot[i]
                        print ('flex_1_3_maximum')
        elif body ==self.Body_2:
            if self.Body_1 != None:
                rot_1 = self.Body_2.rotation_matrix
            else:
                rot_1 = Quaternion([1.0,0.0,0.0,0.0])
            if self.Body_2 != None:
                rot_2 = self.Body_1.rotation_matrix
            else:
                rot_2 = Quaternion([1.0,0.0,0.0,0.0])
            if rot_2 != rot_1:
                rot = rot_1.inverted() @ rot_2
                res = rot.to_euler('XYZ')
                #res = Euler([0.0,0.0,0.0])
            else:
                res = Euler([0.0,0.0,0.0])
            if self.flex:
                print ('flex', res)
                for i in range (len(res)):
                    #res[i] *= -1
                    if self.min_limit_rot[i]*-1 >= res[i] >= self.max_limit_rot[i]*-1  :
                        res[i] = 0.0
                        print ('flex_2_1')
                    elif res[i] > self.min_limit_rot[i]*-1:
                        res[i] -= self.min_limit_rot[i]
                        print ('flex_2_2_minimum')
                    elif res[i] < self.max_limit_rot[i]*-1:
                        res[i] += self.max_limit_rot[i]
                        print ('flex_2_3_maximum')                    
        for i in range(3):
            res[i] = _normalize_angle(res[i])
        return res
class BallJoint (Joint):
    def __init__(self, Body_1,Body_2=None, strength = 1.0,friction = 0.5, pilot_1= [0.0,0.0,0.0], pilot_2 =[0.0,0.0,0.0], local = True, flex = False,
                collision =False, max_length = 0.0,min_length =0.0):
        super().__init__(Body_1=Body_1,Body_2=Body_2, strength = 1.0,friction = 0.5, local = local, flex = flex,
                collision =collision, max_limit_rot = Euler([8.0,8.0,8.0]),min_limit_rot = Euler([-8.0,-8.0,-8.0]),max_length =max_length,min_length =min_length)
class CollisionJoint (Joint):
    def __init__(self, Body_1,Body_2=None, strength = 1.0,friction = 0.5, pilot_1= [0.0,0.0,0.0], pilot_2 =[0.0,0.0,0.0], local = True, flex = False,
                collision =False, max_limit_rot = Euler([0.0,0.0,0.0]),min_limit_rot = Euler([0.0,0.0,0.0]),max_length = 0.0):
        super().__init__(Body_1=Body_1,Body_2=Body_2, strength = 1.0,friction = 0.5, local = local, flex = flex,
                collision =True, max_limit_rot = max_limit_rot,min_limit_rot = min_limit_rot,max_length = 0.0)
    def find_reaction(self):
        if Body_1 != None:
            mass_1 = Body_1.mass
        else:
            mass_1 = -1
        if Body_2!= None:
            mass_2 = Body_2.mass
        else:
            mass_2 = -1
        if (mass_1 > 0.00001 and mass_2 > 0.00001) or (mass_1 < -0.00001 and mass_2 < -0.00001):
            strength_1 = strength* mass_2/mass_1
            strength_2 = strength* mass_1/mass_2
        elif mass_1 < -0.00001 and mass_2 > 0.00001:
            strength_1 = 0
            strength_2 = strength
        elif mass_1 > 0.00001 and mass_2 < -0.00001:
            strength_1 = strength
            strength_2 = 0
        else:
            strength_1 = strength
            strength_2 = strength
        return strength_1,strength_2
class CollisionField:
    def __init__(self, types = 'NONE'):
        global DEBUG
        self.ID = start_time - time.time()
        if DEBUG:
                print ('Colision field',self.ID)
        if types in ['NONE','BULLETS','STATICS','MAGIC','BODY','HAIR', 'PATICULES']:
            self.types = types
        else:
            self.types = 'NONE'
        self.scale_layers = np.array()
class PhysicBody:
    def __init__(self, linked_object = None, length=1.0, width=1.0,height=1.0,mass = 1.0,static = False, new_motion = False):
        global DEBUG
        self.ID = start_time - time.time()
        if DEBUG:
                print ('Physics object',self.ID)
        self.linked_object = linked_object
        self.new_motion = new_motion

        self.length = length
        self.width = width
        self.height = height
        self.mass = mass
        if linked_object != None and new_motion == False:
            self.matrix_base = self.linked_object.matrix_world
            if 'velocity' in self.linked_object:
                self.velocity = Vector(self.linked_object['velocity'])
                self.rel_velocity = Vector(self.linked_object['rel_velocity'])
                self.angular_velocity = Euler(self.linked_object['angular_velocity'])
                self.rel_angular_velocity = Euler(self.linked_object['rel_angular_velocity'])
            else:
                self.linked_object['velocity'] = Vector([0.0,0.0,0.0])
                self.linked_object['rel_velocity'] = Vector([0.0,0.0,0.0])
                self.linked_object['angular_velocity'] = Euler([0.0,0.0,0])
                self.linked_object['rel_angular_velocity'] = Euler([0.0,0.0,0.0])
                if DEBUG:
                    self.linked_object['step'] = 0.0
                self.velocity = Vector([0.0,0.0,0.0])
                self.rel_velocity = Vector([0.0,0.0,0.0])
                self.angular_velocity = Euler([0.0,0.0,0])
                self.rel_angular_velocity = Euler([0.0,0.0,0.0])
        else:
            self.matrix_base = M0
            self.velocity = Vector([0.0,0.0,0.0])
            self.rel_velocity = Vector([0.0,0.0,0.0])
            self.angular_velocity = Vector([0.0,0.0,0])
            self.rel_angular_velocity = Vector([0.0,0.0,0.0])
        self.forces = []#Force(offset= [1.0,0.0,0.0],strength = 1.0,local=True),Force([0.0,0.0,-1.0],strength = 1.0,offset= [-1.0,0.0,0.0],local=True)]
        self.joints = []
        self.accelerations = []
    #@property
    #def matrix(self):
     #   return self.matrix_base
    @property
    def position(self):
        return Vector([self.matrix_base[0][3],self.matrix_base[1][3],self.matrix_base[2][3]])
    @property
    def rotation_euler(self):
        return self.matrix_base.to_euler('XYZ')
    @property
    def rotation_quaternion(self):
        return self.matrix_base.to_quaternion()
    @property
    def rotation_matrix(self):
        return self.matrix_base.to_3x3()
    def set_position(self, pos= [0.0,0.0,0.0]):
        self.matrix_base[0][3] = pos[0]
        self.matrix_base[1][3] = pos[1]
        self.matrix_base[2][3] = pos[2]
    def set_rotation(self, rot= [0.0,0.0,0.0]):
            possible = False
            order = 'xyz'
            if len (rot) ==3:
                print ('euler',rot)
                #r = M0
                r = Euler(rot).to_matrix()
                #for i in range (3):
                #    for j in range (3):
                #        r [i][j] = m [i][j]
                print ('euler',r)
                possible = True
            elif len (rot) ==4:
                r = Quaternion(rot).rotation_matrix
                print ('r',r)
                possible = True
            if possible:
                for i in range (3):
                    for j in range (3):
                        self.matrix_base[i][j] = r [i][j]
    def set_velocity (self,velocity):
        if velocity != None:
            self.velocity =Vector( velocity)
    def set_rel_velocity (self,velocity):
        if velocity != None:
            self.rel_velocity = Vector(velocity)
    def set_angular_velocity (self,velocity):
        if velocity != None:
            self.angular_velocity = Euler(velocity)
    def set_rel_angular_velocity (self,velocity):
        if velocity != None:
            self.rel_angular_velocity = Euler(velocity)
    def add_force(self,force):
        if force!= None:
            self.forces.append (force)
    def clear_forces (self):
        for f in self.forces:
            del f
        self.forces = []
    def remove_force (self ,force=None, num = 0):
        if force == None:
            if len(self.forces) >num:
                del self.forces[num]
                return True
        else:
            vic = False
            for i in range (len(self.forces)):
                if self.forces[i] ==force:
                    del self.forces[i]
                    vic = True
                    break
            if vic:
                return True
    def add_joint(self,joint):
        if joint!= None:
            self.joints.append (joint)
    def clear_joints (self):
        for j in self.joints:
            del j
        self.joints = []
    def remove_joint (self ,joint = None, num = 0):
        if joint == None:
            if len(self.joints) >num:
                del self.joints[num]
                return True
        else:
            vic = False
            for i in range (len(self.joints)):
                if self.joints[i] ==joint:
                    del self.joints[i]
                    vic = True
                    break
            if vic:
                return True
    def apply_motion(self):
        if self.linked_object != None:
            self.linked_object.matrix_world = self.matrix_base
            self.linked_object['velocity'] =  self.velocity
            self.linked_object['rel_velocity'] =  self.rel_velocity
            self.linked_object['angular_velocity'] = self.angular_velocity
            self.linked_object['rel_angular_velocity'] = self.rel_angular_velocity#'''
    def slerp_motion(self,delta=0.01):
        matrix_new = self.matrix_base.copy()
        way = Vector([0.0,0.0,0.0])
        last_rot = Euler([0.0,0.0,0.0]) # Використовується для перевірки в кінці
        
        # --- 1. Ініціалізація сил і моментів ---
        # Ми будемо накопичувати всі сили тут
        total_local_force = Vector([0.0, 0.0, 0.0])
        total_local_torque = Euler([0.0, 0.0, 0.0]) # Ви використовуєте Euler для моментів

        # Переконайтеся, що маса встановлена!
        if self.mass <= 0.0001:
            print(f"Warning: Object {self.linked_object.name} has zero or negative mass!")
            # Використовуємо 1.0, щоб уникнути ділення на нуль
            effective_mass = 1.0
        else:
            effective_mass = self.mass
            
        # Спрощена інерція (для стабільності кутового демпфування)
        # У реальному фреймворку тут має бути тензор інерції
        I_approx = effective_mass 

        # --- 2. Розрахунок сил від Joint (Пружини та Демпфери) ---
        for joint in self.joints:
            # === Лінійна Пружина-Демпфер ===
            
            # Отримуємо поточну швидкість (глобальну)
            res_velocity = self.velocity + self.rel_velocity @ self.rotation_matrix
            
            # Отримуємо вектор помилки (відхилення пружини)
            # j_vec = (target_pos - current_pos)
            j_vec = joint.get_distance(body = self) 
            
            # --- Розрахунок правильних коефіцієнтів ---
            k = joint.strength           # k (Жорсткість пружини)
            zeta = joint.stiffness       # zeta (Ваш коефіцієнт демпфування 0-1)
            
            # Формули для демпфера
            c_crit = 2.0 * sqrt(effective_mass * k)
            c = zeta * c_crit            # c (Реальний коефіцієнт гасіння)

            # --- Обчислення сил ---
            # F = F_spring + F_damp = (-k * x) + (-c * v)
            # Примітка: Ваш get_distance, здається, повертає (target - current), тому ми множимо на k, а не -k
            
            F_spring = k * j_vec
            
            # Гасіння протидіє поточній швидкості
            F_damp = -c * res_velocity 
            
            # Загальна сила від з'єднання (у глобальних координатах)
            F_total_global = F_spring + F_damp
            
            # Конвертуємо у локальну силу для додавання до rel_velocity (як у вашому коді)
            F_total_local = F_total_global @ self.rotation_matrix.transposed()
            total_local_force += F_total_local
            
            # === Кутова Пружина-Демпфер ===

            # Поточна кутова швидкість (локальна, як ви її використовуєте)
            current_ang_vel = self.rel_angular_velocity
            
            # Помилка обертання (відхилення)
            drot = joint.get_rotation_difference(self) # theta
            
            # --- Розрахунок коефіцієнтів ---
            k_rot = joint.torque       # k_theta (Кутова жорсткість)
            zeta_rot = joint.friction  # zeta_theta (Ваш коефіцієнт 0-1)
            
            c_rot_crit = 2.0 * sqrt(I_approx * k_rot)
            c_rot = zeta_rot * c_rot_crit # c_theta (Кутовий коефіцієнт гасіння)

            # --- Обчислення моментів (Torque) ---
            # Tau = Tau_spring + Tau_damp = (-k * theta) + (-c * omega)
            
            Tau_spring = Euler([0.0, 0.0, 0.0])
            Tau_damp = Euler([0.0, 0.0, 0.0])
            
            for i in range(3):
                # Ваш get_rotation_difference, здається, повертає (target-current), тому +
                Tau_spring[i] = k_rot * drot[i] 
                Tau_damp[i] = -c_rot * current_ang_vel[i]
            
            # Зберігаємо 'last_rot' для вашої логіки в кінці
            for i in range (3):
                last_rot[i]+= Tau_spring[i] * delta / I_approx 
            
            # Додаємо загальний момент (локальний)
            for i in range (3):
                total_local_torque[i] += (Tau_spring[i] + Tau_damp[i])

        # --- 3. Розрахунок сил від зовнішніх Forces ---
        # (Ця частина коду виглядає складною, я залишу її, але
        # рекомендую спростити її в майбутньому)
        
        forces = [ f for f in self.forces]
        
        # Додаємо об'єднану силу від Joint як одну локальну силу
        if total_local_force.length > 0.001:
             forces.append(Force(total_local_force, strength=1.0, local=True))
        
        # (Ваша логіка обробки 'forces' та 'moments' залишилася без змін)
        if len (forces) >0:
            if len (forces)==1:
                force = forces[0]
                if force.local:
                    pass
                    self.rel_velocity = self.rel_velocity + force.get_force_vector * delta / effective_mass
                else:
                    self.velocity = self.velocity + force.get_force_vector * delta / effective_mass
                
                # ... (решта вашого коду) ...
                res_velocity = self.velocity + self.rel_velocity @ self.rotation_matrix
                res_angular_velocity= Euler ([0.0,0.0,0.0])
                
                local_angular_velocity = np.dot(self.rotation_matrix.inverted(), self.angular_velocity).flatten()
                for i in range (len(res_angular_velocity)):
                        res_angular_velocity[i]+= self.rel_angular_velocity[i]+local_angular_velocity[i]
                way = res_velocity*delta*1.0
            else:
                res_force = Force([0.0,0.0,0.0])
                res_moment = Euler([0.0,0.0,0.0])
                for force in forces:
                    res_force.add_force_vector(force.get_force_vector)
                    #pass
                for i in range (len(forces)):
                    for j in range (len(forces)-1):
                        if i != j:
                            f0 = forces[i]
                            f1 = forces[j]
                            a = f0.offset - f1.offset
                            b = f1.offset - f0.offset
                            m0 = a.cross(f0.get_force_vector)
                            m1 = b.cross(f1.get_force_vector)
                            res_moment = res_moment+m0
                            res_moment = res_moment+m1
                            print ('Moments',m0,m1)
                scalar = (res_moment[0]**2+ res_moment[1]**2+res_moment[2]**2)**0.5
                if scalar >0.001:
                    #pass
                    #self.angular_velocity =
                    self.angular_velocity = self.angular_velocity + res_moment*delta / I_approx
                print (res_force.get_force_vector)
                if res_force.local:
                    #pass
                    self.rel_velocity = self.rel_velocity + res_force.get_force_vector*delta / effective_mass
                else:
                    self.velocity = self.velocity + res_force.get_force_vector*delta / effective_mass
                
                # ... (решта вашого коду) ...
                res_velocity = self.velocity +self.rel_velocity @ self.rotation_matrix
                res_angular_velocity= Euler ([0.0,0.0,0.0])
                local_angular_velocity = np.dot(self.rotation_matrix.inverted(), self.angular_velocity).flatten()
                for i in range (len(res_angular_velocity)):
                        res_angular_velocity[i]+= self.rel_angular_velocity[i]+local_angular_velocity[i]
                way = res_velocity*delta
        else:
            res_velocity = self.velocity +self.rel_velocity @ self.rotation_matrix
            res_angular_velocity= Euler ([0.0,0.0,0.0])
            local_angular_velocity = np.dot(self.rotation_matrix.inverted(), self.angular_velocity).flatten()
            for i in range (len(res_angular_velocity)):
                        res_angular_velocity[i]+= self.rel_angular_velocity[i]+local_angular_velocity[i]
            way = res_velocity*delta

        # --- 4. Інтеграція кутової швидкості (від Joint) ---
        # Додаємо накопичений момент до кутової швидкості
        # a_rot = Torque / Inertia
        for i in range (3):
            self.rel_angular_velocity[i] += (total_local_torque[i] * delta / I_approx)
        
        # Оновлюємо res_angular_velocity новим значенням
        for i in range (len(res_angular_velocity)):
            res_angular_velocity[i] = self.rel_angular_velocity[i] + local_angular_velocity[i]

        # --- 5. Інтеграція позиції та обертання (ваш код) ---
        rot_1 =Euler ([0.0,0.0,0.0])
        for i in range (3):
            rot_1[i] = res_angular_velocity[i]*delta
        
        matrix_new = (self.rotation_matrix @ (rot_1).to_matrix()).normalized()
        for i in range (3):
            for j in range (3):
                self.matrix_base[i][j] = matrix_new[i][j]
            self.matrix_base[i][3] += way[i]
                
        return self.matrix_base
A = PhysicBody(bpy.data.objects['Cube'])
B = PhysicBody(bpy.data.objects['Cube.001'])
print (A,B)
J= Joint (B,A, pilot_1= [0.0,0.0,-1.5], pilot_2 =[0.0,0.0,1.50],flex = True,strength = 10.0,torque = 1.0,friction = 0.9,max_limit_rot = Euler([pi/2,pi/2,pi/2]),min_limit_rot = Euler([-pi/4,-pi/4,-pi/4]))
A.joints.append (J)
B.joints.append (J)
for i in range (20):
    for body in [A,B]:
        body.slerp_motion(0.005)
    for body in [A,B]:
        body.apply_motion ()
