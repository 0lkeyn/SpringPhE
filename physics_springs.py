import sys , os, time
from math import *
from mathutils import *
import numpy as np
from numpy_da import DynamicArray
from random import random
from copy import copy
import bpy
import bgl
import blf
import concurrent.futures
from bpy.utils import register_class
import uuid

DEBUG = False
COLLISIONS = 0
# === ГЛОБАЛЬНЕ СХОВИЩЕ СИМУЛЯЦІЇ ===
# Тут будуть жити наші об'єкти, поки працює Блендер
SIM_BODIES = []
SIM_FIELD = None


start_time = time.time()
q_zero = Quaternion((1.0,0.0,0.0,0.0))
q_zero0 = Quaternion((-1.0,0.0,0.0,0.0))
M0 = Matrix([[1.0,0.0,0.0,0.0]
                ,[0.0,1.0,0.0,0.0]
                ,[0.0,0.0,1.0,0.0]
                ,[0.0,0.0,0.0,1.0]])
#def _normalize_angle(angle):
#    """Нормалізує кут в діапазон [-pi, pi]."""
#    return atan2(sin(angle), cos(angle))
def _normalize_angle(angle):
    while angle > pi:
        angle -= 2 * pi
    while angle < -pi:
        angle += 2 * pi
    return angle
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

def get_lowest_point_fast(obj,R):
    """
    obj: об'єкт Blender (або ваш PhysicBody)
    dims: Vector([half_x, half_y, half_z]) - половини розмірів куба
    """
    #R = obj.rotation_euler.to_matrix() # Або body.rotation_matrix
    dims = obj.dimensions / 2
    # 1. Глобальний вектор "вниз"
    global_down = Vector((0.0, 0.0, -1.0))
    
    # 2. Переводимо "вниз" у локальні координати (R^T * v)
    # У mathutils: vector @ matrix робить саме це
    local_down = global_down @ R
    
    # 3. Визначаємо знаки (+ або -) для кожного виміру
    # sign(x) повертає 1 якщо x >= 0, інакше -1
    sign_x = 1 if local_down.x >= 0 else -1
    sign_y = 1 if local_down.y >= 0 else -1
    sign_z = 1 if local_down.z >= 0 else -1
    
    # 4. Формуємо локальну координату найнижчої вершини
    # dims - це половини розмірів (наприклад, 1.0 для стандартного куба)
    lowest_local = Vector((
        sign_x * dims[0], 
        sign_y * dims[1], 
        sign_z * dims[2]
    ))
    return lowest_local

def get_lowest_point_fast_global(obj, dims):
    R = obj.rotation_euler.to_matrix()
    lowest_local = get_lowest_point_fast(obj, dims)
    # 5. Переводимо знайдену точку назад у глобальні координати
    # (Обертання + Позиція)
    lowest_global = (R @ lowest_local) + obj.position
    
    return lowest_global

class Force :
    def __init__(self, vector = [0.0,0.0,1.0], strength = 1.0, offset= [0.0,0.0,0.0], local = True, immortal = True, life = 1.0):
        global DEBUG
        self.ID = id(self)
        if DEBUG:
            pass
            #print ('Force object',self.ID)
        self.strength = strength
        self.offset = Vector(offset)
        self.vector = Vector(vector)
        self.local = local
        self.immortal = immortal #if True force had no termination time
        self.life = life #time points to termination force influence
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
    def __init__(self, vector = [0.0,0.0,1.0], strength = 1.0, offset= [0.0,0.0,0.0], local = True, types = 'NONE',immortal = True,life=1.0):
        super().__init__(vector=vector,strength=strength,offset=offset,local=local,immortal=immortal,life=life)
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
    def __init__(self, Body_1,Body_2=None, strength = 1.0, stiffness =  Vector([1.0,1.0,1.0]),torque = 0.5,friction = 0.1, pilot_1= [0.0,0.0,0.0], pilot_2 =[0.0,0.0,0.0], local = True, flex = True, rigid = True,
                collision =False, max_limit_rot = Euler([0.0,0.0,0.0]),min_limit_rot = Euler([0.0,0.0,0.0]),min_length = 0.0,max_length = 10.0,active = True, types = None):
        global DEBUG
        self.ID = id(self)
        self.types = types
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
                offset_1 = self.pilot_1 @ self.Body_1.last_rotation_matrix.transposed()
                pos_1 = self.Body_1.last_position
            else:
                offset_1 = self.pilot_1
                pos_1 = Vector([0.0,0.0,0.0])
            if self.Body_2 != None:
                offset_2= self.pilot_2 @ self.Body_2.last_rotation_matrix.transposed()
                pos_2 = self.Body_2.last_position
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
                offset_1 = self.pilot_1 @ self.Body_2.last_rotation_matrix.transposed()
                pos_1 = self.Body_1.last_position #+ offset_1
            else:
                offset_1 = self.pilot_1
                pos_1 = Vector([0.0,0.0,0.0])
            if self.Body_2 != None:
                offset_2= self.pilot_2@ self.Body_1.last_rotation_matrix.transposed()
                pos_2 = self.Body_2.last_position #+ offset_2
            else:
                offset_2 = self.pilot_2
                pos_2 = Vector([0.0,0.0,0.0])
            vec = pos_1 - pos_2 + offset_1 - offset_2
            if self.rigid:
                return vec
            else:
                if vec.length > self.min_length:
                        if  vec.length > self.max_length:
                            self.active = False
                            return Vector([0.0,0.0,0.0])
                        else:
                            return Vector([0.0,0.0,0.0])
                return vec
                    
    def get_rotation_difference(self, body):
        # 1. Отримуємо кватерніони
        q1 = self.Body_1.rotation_quaternion if self.Body_1 else Quaternion((1,0,0,0))
        q2 = self.Body_2.rotation_quaternion if self.Body_2 else Quaternion((1,0,0,0))

        # 2. Різниця: q_diff * q1 = q2  =>  q_diff = q2 * inv(q1)
        # У локальному просторі Body_1:
        q_diff = q1.conjugated() @ q2 
        
        # Shortest path (щоб не крутило через 360)
        if q_diff.w < 0:
            q_diff = -q_diff

        # 3. Перетворення Quaternion -> Rotation Vector (Axis-Angle)
        # Це найстабільніший метод. Ніяких Euler singularities.
        
        # Кут обертання
        angle = 2.0 * acos(min(max(q_diff.w, -1.0), 1.0))
        
        rot_vector = Vector((0.0, 0.0, 0.0))
        
        if angle > 0.0001:
            sin_val = sqrt(1.0 - q_diff.w * q_diff.w)
            if sin_val < 0.0001: sin_val = 1.0
            
            # Вісь обертання
            axis = Vector((q_diff.x, q_diff.y, q_diff.z)) / sin_val
            
            # Вектор помилки (напрямок = вісь, довжина = кут)
            rot_vector = axis * angle
            
        # 4. Ліміти (якщо Flex)
        correction = Vector((0.0, 0.0, 0.0))
        
        if self.flex:
            for i in range(3):
                val = rot_vector[i]
                min_lim = self.min_limit_rot[i]
                max_lim = self.max_limit_rot[i]
                
                if val > max_lim:
                    correction[i] = val - max_lim
                elif val < min_lim:
                    correction[i] = val - min_lim
                # else correction is 0
        else:
            correction = rot_vector

        # 5. Знак
        # Якщо Body_1, ми хочемо повернути його ДО Body_2 -> correction
        if body == self.Body_1:
            return correction
        # Якщо Body_2, ми хочемо повернути його назад -> -correction
        elif body == self.Body_2:
            return -correction
            
        return Vector((0.0, 0.0, 0.0))
class BallJoint (Joint):
    def __init__(self, Body_1,Body_2=None, strength = 1.0,torque = 0.0,friction = 0.5, pilot_1= [0.0,0.0,0.0], pilot_2 =[0.0,0.0,0.0], local = True, flex = False,
                collision =False, max_length = 1.01,min_length =0.0):
        super().__init__(Body_1=Body_1,Body_2=Body_2, strength = 1.0,torque = 0.0, friction = 0.5, local = local, flex = flex,
                collision =collision, max_limit_rot = Euler([8.0,8.0,8.0]),min_limit_rot = Euler([-8.0,-8.0,-8.0]), max_length = max_length, min_length = min_length, types = 'BALL')
class CollisionJoint (Joint):
    def __init__(self, Body_1,Body_2=None, strength = 1.0,torque = 0.0,stiffness =  Vector([1.0,1.0,1.0]),friction = 0.9, pilot_1= [0.0,0.0,0.0], pilot_2 =[0.0,0.0,0.0], local = True, flex = False, rigid = True,
                collision =False, max_limit_rot = Euler([8.0,8.0,8.0]),min_limit_rot = Euler([-8.0,-8.0,-8.0]),min_length = 0.0,max_length = 0.01):
        super().__init__(Body_1=Body_1,Body_2=Body_2, strength = strength,torque=0.0,friction = 0.9,stiffness=stiffness,pilot_1=pilot_1,pilot_2=pilot_2, local = local, flex = flex, rigid = rigid,
                collision =True, max_limit_rot = max_limit_rot,min_limit_rot = min_limit_rot,min_length = min_length, max_length = max_length, types = 'COLLISION')
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
# --- ДОПОМІЖНІ ФУНКЦІЇ ---

def get_local_corners(dims):
    """Повертає 8 вершин куба відносно його центру (локальні координати)"""
    x, y, z = dims[0]/2, dims[1]/2, dims[2]/2
    return [
        Vector(( x,  y,  z)), Vector(( x,  y, -z)),
        Vector(( x, -y,  z)), Vector(( x, -y, -z)),
        Vector((-x,  y,  z)), Vector((-x,  y, -z)),
        Vector((-x, -y,  z)), Vector((-x, -y, -z))
    ]

def get_box_contact_points(body_a, body_b, margin=0.05):
    """
    Знаходить всі вершини body_a, які проникли в body_b, і навпаки.
    Повертає список: [(point_a_local, point_b_local, normal, depth), ...]
    """
    contacts = []
    
    # Вектор від центру B до центру A (для перевірки напрямку нормалі)
    center_vec = body_a.position - body_b.position
    
    def check_vertices_in_box(source, target, normal_flip=False):
        src_local_corners = get_local_corners(source.dimensions)
        
        # Матриці для переходу між просторами
        src_to_world = source.matrix_base
        world_to_tgt = target.matrix_base.inverted()
        
        # Половини розмірів Target (+ margin)
        tgt_h = target.dimensions / 2.0
        expanded_h = tgt_h + Vector((margin, margin, margin))

        for pt_local in src_local_corners:
            # 1. Точка Source -> World
            pt_global = src_to_world @ pt_local
            
            # 2. World -> Target Local
            pt_in_tgt = world_to_tgt @ pt_global
            
            # 3. Перевірка AABB (в локальному просторі Target)
            if (abs(pt_in_tgt.x) < expanded_h.x and 
                abs(pt_in_tgt.y) < expanded_h.y and 
                abs(pt_in_tgt.z) < expanded_h.z):
                
                # 4. Розрахунок глибини та нормалі
                dx_p = tgt_h.x - pt_in_tgt.x
                dx_n = tgt_h.x + pt_in_tgt.x
                dy_p = tgt_h.y - pt_in_tgt.y
                dy_n = tgt_h.y + pt_in_tgt.y
                dz_p = tgt_h.z - pt_in_tgt.z
                dz_n = tgt_h.z + pt_in_tgt.z
                
                min_depth = min(dx_p, dx_n, dy_p, dy_n, dz_p, dz_n)
                
                # Локальна нормаль (від стінки Target)
                local_normal = Vector((0,0,0))
                if min_depth == dx_p: local_normal = Vector((1,0,0))
                elif min_depth == dx_n: local_normal = Vector((-1,0,0))
                elif min_depth == dy_p: local_normal = Vector((0,1,0))
                elif min_depth == dy_n: local_normal = Vector((0,-1,0))
                elif min_depth == dz_p: local_normal = Vector((0,0,1))
                elif min_depth == dz_n: local_normal = Vector((0,0,-1))
                
                # Глобальна нормаль
                global_normal = target.rotation_matrix @ local_normal
                global_normal.normalize()
                
                # КРИТИЧНО: Нормаль має штовхати тіла нарізно!
                # Перевіряємо, чи дивиться вона в бік body_a (center_vec)
                if center_vec.dot(global_normal) < 0:
                     global_normal = -global_normal
                
                # Записуємо точки
                pt_a_loc = src_to_world.inverted() @ pt_global
                pt_b_loc = pt_in_tgt
                
                if normal_flip:
                    # Якщо перевіряли B в A, то глобальна нормаль розрахована від A.
                    # Але ми хочемо єдиний стандарт (від B до A), тому інвертуємо
                    global_normal = -global_normal 
                    contacts.append((pt_b_loc, pt_a_loc, -global_normal, min_depth))
                else:
                    contacts.append((pt_a_loc, pt_b_loc, global_normal, min_depth))

    # Перевіряємо обидва напрямки для стабільності
    check_vertices_in_box(body_a, body_b, normal_flip=False)
    check_vertices_in_box(body_b, body_a, normal_flip=True)
    
    return contacts

def get_cube_floor_contact_multi(cube, floor, margin=0.05):
    """
    Повертає список контактів для куба і підлоги.
    """
    contacts = []
    plane_normal = floor.matrix_base.to_3x3() @ Vector((0.0, 0.0, 1.0))
    plane_normal.normalize()
    plane_pos = floor.position
    
    corners = get_local_corners(cube.dimensions)
    cube_mat = cube.matrix_base
    
    for local_pt in corners:
        global_pt = cube_mat @ local_pt
        distance = (global_pt - plane_pos).dot(plane_normal)
        
        if distance < margin:
            penetration = margin - distance
            
            # Точки
            pt_a_local = cube_mat.inverted() @ global_pt
            # Проекція на площину
            global_on_plane = global_pt - plane_normal * (penetration - margin)
            pt_b_local = floor.matrix_base.inverted() @ global_on_plane
            
            # Нормаль повертаємо від підлоги вгору (до куба)
            contacts.append((pt_a_local, pt_b_local, plane_normal, penetration))
            
    return contacts


# --- ОНОВЛЕНИЙ КЛАС COLLISIONFIELD ---

class CollisionField:
    def __init__(self, types = 'NONE',bodies = []):
        global DEBUG
        self.ID = id(self)
        self.types = types
        self.bodies = bodies
        self.joints = {}

    def detect_collision(self, delta=0.1):
        margin = 0.1 # Чутливість колізії
        
        # 1. ОЧИЩЕННЯ: Видаляємо всі старі колізійні джойнти
        for body in self.bodies:
            body.joints = [j for j in body.joints if not isinstance(j, CollisionJoint)]
        
        # 2. Функція створення контакту
        def add_contact_point(body_a, body_b, pt_a, pt_b, norm, pen):
             # Створюємо джойнт
             # Зверніть увагу: strength можна зменшити, бо точок тепер багато (наприклад, ділити на кількість точок)
             col_joint = CollisionJoint(
                Body_1=body_a, 
                Body_2=body_b, 
                strength=200.0,        # Сила пружини (K)
                stiffness=Vector((1,1,1)), 
                pilot_1=pt_a,          # Локальна точка на тілі A
                pilot_2=pt_b,          # Локальна точка на тілі B
                min_length=0.0, 
                max_length=pen + 0.1,  # Довжина спокою
                friction=0.5           # Тертя
             )
             # Важливо: torque > 0 дає обертання при ударі кутом
             col_joint.torque = 5.0 
             
             body_a.joints.append(col_joint)
             body_b.joints.append(col_joint)

        # 3. Перебір пар тіл
        n = len(self.bodies)
        for i in range(n):
            for j in range(i + 1, n):
                a = self.bodies[i]
                b = self.bodies[j]
                
                # AABB (Bound Box) Optimisation
                dist = (a.position - b.position).length
                radius_sum = max(a.dimensions) + max(b.dimensions)
                if dist > radius_sum: continue

                # BOX vs BOX
                if a.geometry == 'BOX' and b.geometry == 'BOX':
                    contacts = get_box_contact_points(a, b, margin=margin)
                    
                    if contacts:
                        max_pen = 0.0
                        best_normal = Vector((0,0,0))
                        
                        # Додаємо джойнти для ВСІХ знайдених точок (до ліміту, наприклад 8)
                        for pt_a, pt_b, normal, pen in contacts:
                            add_contact_point(a, b, pt_a, pt_b, normal, pen)
                            if pen > max_pen:
                                max_pen = pen
                                best_normal = normal
                        
                        # М'яка корекція позиції (Anti-Jitter)
                        # correction_factor = 0.4 (не виштовхуємо миттєво)
                        if max_pen > 0:
                            correction = best_normal * max_pen * 0.4
                            if not a.is_static and not b.is_static:
                                a.matrix_base.translation += correction
                                b.matrix_base.translation -= correction
                            elif not a.is_static:
                                a.matrix_base.translation += correction * 2
                            elif not b.is_static:
                                b.matrix_base.translation -= correction * 2

                # BOX vs FLOOR
                elif (a.geometry == 'BOX' and b.geometry == 'FLOOR') or \
                     (b.geometry == 'BOX' and a.geometry == 'FLOOR'):
                    
                    # Нормалізуємо: A=Box, B=Floor
                    box, floor = (a, b) if a.geometry == 'BOX' else (b, a)
                    
                    contacts = get_cube_floor_contact_multi(box, floor, margin=margin)
                    
                    if contacts:
                        max_pen = 0.0
                        best_normal = Vector((0,0,1))
                        
                        for pt_a, pt_b, normal, pen in contacts:
                            # normal дивиться від підлоги до куба
                            add_contact_point(box, floor, pt_a, pt_b, normal, pen)
                            if pen > max_pen:
                                max_pen = pen
                                best_normal = normal
                        
                        # Корекція позиції (штовхаємо куб по нормалі)
                        if not box.is_static and max_pen > 0:
                            box.matrix_base.translation += best_normal * max_pen * 0.4

        #print (n)                    
class PhysicBody:
    def __init__(self, linked_object = None, dimensions = Vector([2.0,2.0,2.0]),mass = 1.0,is_static = False, new_motion = False, collision = 'DEFAULT',geometry = 'BOX', collision_layer = 0, sticky = -0.60, last_collision = {}):
        global DEBUG
        self.ID = id(self)
        if DEBUG:
                print ('Physics object',self.ID)
        self.linked_object = linked_object
        self.new_motion = new_motion
        if linked_object != None:
            self.dimensions = Vector(linked_object.dimensions)
        else:
            self.dimensions = Vector(dimensions)
        self.length = dimensions.x
        self.width = dimensions.y
        self.height = dimensions.z
        self.mass = mass
        self.gravity = Vector((0.0,0.0,-9.8))
        if collision in ['DEFAULT', 'NONE', 'STATIC', 'ONLY_LAYER', 'NOT_LAYER']:
            self.collision = collision
            self.collision_layer = collision_layer
        else:
            self.collision = 'NONE'
            self.collision_layer = 0
        if geometry in ['BOX','SPHERE','PLANE','FLOOR','POINT','TERRAIN','BULLET','CYLINDER','CAPSULE','MESH']:
            self.geometry = geometry
        else:
            self.geometry = 'NONE'
        self.sticky = sticky
        self.is_static = is_static
        if linked_object != None and new_motion == False:
            self.matrix_base = self.linked_object.matrix_world.copy()
            self.last_matrix_base = self.linked_object.matrix_world.copy()
            if DEBUG:
                print ('M',self.matrix_base)
            if 'velocity' in self.linked_object:
                self.velocity = Vector(self.linked_object['velocity'])
                self.rel_velocity = Vector(self.linked_object['rel_velocity'])
                self.angular_velocity = Vector(self.linked_object['angular_velocity'])
                self.rel_angular_velocity = Vector(self.linked_object['rel_angular_velocity'])
                self.last_velocity = Vector(self.linked_object['velocity'])
                self.last_rel_velocity = Vector(self.linked_object['rel_velocity'])
                self.last_angular_velocity = Vector(self.linked_object['angular_velocity'])
                self.last_rel_angular_velocity = Vector(self.linked_object['rel_angular_velocity'])
            else:
                self.linked_object['velocity'] = Vector([0.0,0.0,0.0])
                self.linked_object['rel_velocity'] = Vector([0.0,0.0,0.0])
                self.linked_object['angular_velocity'] = Vector([0.0,0.0,0])
                self.linked_object['rel_angular_velocity'] = Vector([0.0,0.0,0.0])
                if DEBUG:
                    self.linked_object['step'] = 0.0
                self.velocity = Vector([0.0,0.0,0.0])
                self.rel_velocity = Vector([0.0,0.0,0.0])
                self.angular_velocity = Vector([0.0,0.0,0])
                self.rel_angular_velocity = Vector([0.0,0.0,0.0])
                self.last_velocity = Vector([0.0,0.0,0.0])
                self.last_rel_velocity = Vector([0.0,0.0,0.0])
                self.last_angular_velocity = Vector([0.0,0.0,0])
                self.last_rel_angular_velocity = Vector([0.0,0.0,0.0])
        else:
            self.matrix_base = M0
            self.lasrt_matrix_base = M0
            self.velocity = Vector([0.0,0.0,0.0])
            self.rel_velocity = Vector([0.0,0.0,0.0])
            self.angular_velocity = Vector([0.0,0.0,0])
            self.rel_angular_velocity = Vector([0.0,0.0,0.0])
            self.last_velocity = Vector([0.0,0.0,0.0])
            self.last_rel_velocity = Vector([0.0,0.0,0.0])
            self.last_angular_velocity = Vector([0.0,0.0,0])
            self.last_rel_angular_velocity = Vector([0.0,0.0,0.0])
        self.forces = []#Force(offset= [1.0,0.0,0.0],strength = 1.0,local=True),Force([0.0,0.0,-1.0],strength = 1.0,offset= [-1.0,0.0,0.0],local=True)]
        self.joints = []        
        self.collisions ={}
        self.accelerations = []#Acceleration(vector = [0.0,0.0,-1.0], strength = 0.98, offset= [0.0,0.0,0.0], local = False, types = 'GRAVITY')]
        self.recorder = []
    
    @property
    def position(self):
        return Vector([self.matrix_base[0][3],self.matrix_base[1][3],self.matrix_base[2][3]])
    @property
    def last_position(self):
        return Vector([self.last_matrix_base[0][3],self.last_matrix_base[1][3],self.last_matrix_base[2][3]])
    @property
    def rotation_euler(self):
        return self.matrix_base.to_euler('XYZ')
    @property
    def last_rotation_euler(self):
        return self.last_matrix_base.to_euler('XYZ')
    @property
    def rotation_quaternion(self):
        return self.matrix_base.to_quaternion()
    @property
    def last_rotation_quaternion(self):
        return self.last_matrix_base.to_quaternion()
    @property
    def rotation_matrix(self):
        return self.matrix_base.to_3x3()
    @property
    def last_rotation_matrix(self):
        return self.last_matrix_base.to_3x3()
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
            self.angular_velocity = Vector(velocity)
    def set_rel_angular_velocity (self,velocity):
        if velocity != None:
            self.rel_angular_velocity = Vector(velocity)
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
    # 1. Новий метод (Додайте в клас PhysicBody)
    def validate_joints(self):
        """Перевіряє довжину пружин і маркує їх як неактивні ДО фізичного кроку"""
        for joint in self.joints:
            if not joint.active:
                continue
            
            if hasattr(joint, 'max_length') and joint.max_length:
                if isinstance(joint.max_length, (float, int)) and joint.max_length > 0.001:
                    limit = joint.max_length
                elif isinstance(joint.max_length, Vector) and joint.max_length.length > 0.001:
                    limit = joint.max_length.length
                else:
                    continue
                
                if joint.Body_1 and joint.Body_2:
                    # ВИПРАВЛЕНО: правильний порядок множення та використання last_position
                    p1 = joint.Body_1.last_position + (joint.Body_1.last_rotation_matrix @ joint.pilot_1)
                    p2 = joint.Body_2.last_position + (joint.Body_2.last_rotation_matrix @ joint.pilot_2)
                    dist = (p1 - p2).length
                    
                    if dist > limit:
                        joint.active = False # Маркуємо на видалення, але ще не видаляємо
    def sync_to_blender(self):
        if self.linked_object != None:
            self.last_matrix_base = self.matrix_base.copy()
            self.linked_object.matrix_world = self.matrix_base.copy()
            self.last_velocity =  self.velocity
            self.last_rel_velocity =  self.rel_velocity
            self.last_angular_velocity = self.angular_velocity
            self.last_rel_angular_velocity = self.rel_angular_velocity
            self.linked_object['velocity'] =  self.velocity
            self.linked_object['rel_velocity'] =  self.rel_velocity
            self.linked_object['angular_velocity'] = self.angular_velocity
            self.linked_object['rel_angular_velocity'] = self.rel_angular_velocity#'''
            return True
    def slerp_motion(self, delta=0.01):
        if self.is_static: return self.matrix_base
        R = self.last_rotation_matrix 
        
        # Сили - Глобальні
        total_global_force = Vector([0.0, 0.0, 0.0])
        # Моменти - ЛОКАЛЬНІ (Зміна!)
        total_local_torque = Vector([0.0, 0.0, 0.0])
        
        # Гравітація
        if self.mass > 0.0001:
            total_global_force += self.gravity * self.mass
            effective_mass = self.mass
        else:
            effective_mass = 1.0

        # Момент інерції (Локальний)
        I_approx = Vector([1.0, 1.0, 1.0]) * effective_mass * 2.0

        for joint in self.joints:
            if not joint.active: continue
            # Визначаємо ролі
            if joint.Body_1 == self:
                target_body = joint.Body_2
                my_pilot = joint.pilot_1
                other_pilot = joint.pilot_2
            else:
                target_body = joint.Body_1
                my_pilot = joint.pilot_2
                other_pilot = joint.pilot_1

            # === ЛІНІЙНІ СИЛИ (Global Calculation) ===
            # (Тут код майже без змін, бо для позиції Global - це ок)
            
            my_anchor_global = self.last_position + (my_pilot @ R.transposed())
            
            # Обчислення швидкості точки (враховуючи локальну кутову, переведену в глобал)
            omega_global = self.last_rel_angular_velocity @ R.transposed()
            r_offset = my_anchor_global - self.last_position
            my_point_vel = self.velocity + omega_global.cross(r_offset)

            if target_body != None:
                other_R = target_body.last_rotation_matrix
                other_anchor_global = target_body.last_position + (other_pilot @ other_R.transposed())
                other_omega = target_body.last_rel_angular_velocity @ other_R.transposed()
                other_r_offset = other_anchor_global - target_body.last_position
                other_point_vel = target_body.last_velocity + other_omega.cross(other_r_offset)
            else:
                other_anchor_global = Vector(other_pilot) # Static world point
                other_point_vel = Vector((0.0, 0.0, 0.0))

            # Вектори пружини та швидкості
            spring_vec_global = other_anchor_global - my_anchor_global
            '''if joint.rigid == False:
                    if spring_vec_global.length > joint.min_length:
                        if spring_vec_global.length > joint.max_length+delta*1.0:
                            joint.active = False                            
                    else:
                            spring_vec_global =  Vector((0.0, 0.0, 0.0))'''
            
            #spring_vec_global = joint.get_distance (self)
            rel_vel_global = my_point_vel - other_point_vel
            
            # Переводимо в Local для Stiffness
            spring_vec_local = spring_vec_global @ R
            rel_vel_local = rel_vel_global @ R
            
            force_local = Vector([0.0, 0.0, 0.0])

            # Лінійна пружина + Демпфер
            for i in range(3):
                k = joint.strength
                if not isinstance(k, (float, int)): k = k[i]
                zeta = joint.stiffness[i]
                
                if k <= 0.0001: continue
                
                f_spring = k * spring_vec_local[i]
                
                c_crit = 2.0 * sqrt(effective_mass * k)
                f_damp = -c_crit * zeta * rel_vel_local[i]
                
                # Clamping
                max_damp = abs(rel_vel_local[i] * effective_mass / delta)
                if abs(f_damp) > max_damp: f_damp = copysign(max_damp, f_damp)

                force_local[i] = f_spring + f_damp

            # Force Local -> Force Global
            f_global_joint = force_local @ R.transposed()
            total_global_force += f_global_joint
            
            # Torque від лінійної сили (r x F) - це Global Torque
            torque_from_force_global = r_offset.cross(f_global_joint)
            
            # ОДРАЗУ переводимо цей момент у ЛОКАЛЬНИЙ і додаємо
            # Global Torque @ Matrix = Local Torque
            total_local_torque += torque_from_force_global @ R


            # === КУТОВА ПРУЖИНА (Strictly Local) ===
            # Ми більше не рахуємо кватерніони тут вручну.
            # Ми довіряємо joint.get_rotation_difference, який повертає ВЕКТОР ПОМИЛКИ.
            
            # Перевірка на наявність torque
            t_val = joint.torque
            if hasattr(t_val, 'length'): t_val = t_val.length # Якщо вектор
            
            if abs(t_val) > 0.001:
                # Отримуємо вектор помилки в ЛОКАЛЬНИХ координатах (Rotation Vector)
                rot_error_local = joint.get_rotation_difference(self)
                
                # Момент пружини: T = k * angle_vector
                # (Тут припускаємо скалярний torque для простоти, або покомпонентно)
                torque_spring_local = rot_error_local * joint.torque
                
                # Демпфування обертання: T = -c * omega_local
                # Використовуємо friction як коефіцієнт
                torque_damp_local = -self.rel_angular_velocity * joint.friction
                
                total_local_torque += torque_spring_local + torque_damp_local


        # === ІНТЕГРАЦІЯ ===
        
        LINEAR_DRAG = 0.99
        ANGULAR_DRAG = 0.95

        # 1. Лінійна (Global)
        self.velocity += (total_global_force / effective_mass) * delta
        self.velocity *= LINEAR_DRAG
        
        # 2. Кутова (Local) -> БЕЗ перетворень координат!
        # alpha = torque / I
        alpha = Vector((
            total_local_torque.x / I_approx.x,
            total_local_torque.y / I_approx.y,
            total_local_torque.z / I_approx.z
        ))
        
        self.rel_angular_velocity += alpha * delta
        self.rel_angular_velocity *= ANGULAR_DRAG

        # 3. Update Position
        self.matrix_base.translation += self.velocity * delta

        # 4. Update Rotation (Quaternion Integration)
        omega = self.rel_angular_velocity # Це вже Vector
        if omega.length > 0.00001:
            q_curr = self.matrix_base.to_quaternion()
            
            # Створюємо малий поворот з вектора кутової швидкості
            theta = omega.length * delta
            axis = omega.normalized()
            q_delta = Quaternion(axis, theta)
            
            # Інтегруємо: New = Old @ Delta (Local update)
            # Якщо omega - це локальна швидкість обертання навколо власних осей
            q_new = q_curr @ q_delta 
            q_new.normalize()
            
            # Записуємо назад
            loc = self.matrix_base.translation.copy()
            self.matrix_base = q_new.to_matrix().to_4x4()
            self.matrix_base.translation = loc
        
        return self.matrix_base
    def record_frame(self, frame_num):
        """Зберігає поточний стан у пам'ять"""
        if not self.is_static:
            # Зберігаємо копію матриці, щоб вона не змінювалася в майбутньому
            self.recorder.append((frame_num, self.matrix_base.copy()))

class PhysicsSetupOperator(bpy.types.Operator):
    """Створити фізичні тіла з виділених об'єктів і запустити"""
    bl_idname = "physics.setup_and_run"
    bl_label = "Setup & Run Physics"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        global SIM_BODIES, SIM_FIELD
        
        selected_objs = context.selected_objects
        if not selected_objs:
            self.report({'WARNING'}, "Нічого не виділено!")
            return {'CANCELLED'}

        # 1. Очищення попередньої симуляції
        SIM_BODIES.clear()
        SIM_FIELD = None
        
        print(f"--- Initializing Physics for {len(selected_objs)} objects ---")

        # 2. Створення PhysicBody для кожного об'єкта
        for obj in selected_objs:
            # Спроба дістати масу з властивостей об'єкта (Custom Properties)
            # Якщо виставити в об'єкті властивість "mass" = 0, він буде статичним
            mass = obj.get("mass", 1.0) 
            geometry_type = obj.get("geometry",'BOX')
            collision_type = obj.get("collision",'DEFAULT')
            # Визначаємо, чи це статика
            is_static = (obj.dimensions.z <= 0.0001) or (mass <= 0.0001)
            if is_static:
                collision_type = 'STATIC'
                #geometry_type = 'FLOOR'
            else:
                collision_type = 'DEFAULT' # Можна теж читати з властивостей obj.get("collision", 'BOX')
                #geometry_type = 'BOX'
            

            # Створення екземпляра
            pb = PhysicBody(
                linked_object=obj, 
                mass=mass, 
                is_static=is_static,
                collision=collision_type,
                geometry = geometry_type
            )
            
            # Ініціалізація матриць (важливо для e005)
            # (Якщо це не зроблено в __init__, робимо тут примусово)
            pb.matrix_base = obj.matrix_world.copy()
            pb.last_matrix_base = obj.matrix_world.copy()
            SIM_BODIES.append(pb)
            print(f"Added: {obj.name}, Mass: {mass}, Static: {is_static}")

        # 3. Створення CollisionField
        # Передаємо всі тіла, щоб вони могли стикатися
        SIM_FIELD = CollisionField(bodies=SIM_BODIES)
        
        # 4. Автоматичний запуск модального оператора
        #bpy.ops.wm.physics_modal_operator()
        
        return {'FINISHED'}
class PhysicsModalOperator(bpy.types.Operator):
    """Інтерактивна фізика: Запис у пам'ять -> Запікання в кінці"""
    bl_idname = "wm.physics_modal_operator"
    bl_label = "Start Simulation Loop"
    
    _is_running = False
    _timer = None
    _executor = None
    
    # Налаштування
    substeps = 10
    fps = 24.0
    dt_frame = 1.0 / fps
    dt_sub = dt_frame / substeps
    
    # Поточний кадр запису
    _current_rec_frame = 1.0
    animation_active = True
    @classmethod
    def poll(cls, context):
        return not cls._is_running

    def modal(self, context, event):
        if event.type == 'ESC':
            # Користувач натиснув ESC - зупиняємо і ЗАПІКАЄМО
            return self.cancel(context)

        if event.type == 'TIMER':
            self.step_physics()
            # Оновлюємо вьюпорт для візуалізації
            # (Можна закоментувати, якщо хочете супер-швидкість без картинки)
            context.area.tag_redraw()

        return {'PASS_THROUGH'}

    def step_physics(self):
        global SIM_BODIES, SIM_FIELD
        
        if not SIM_BODIES: return

        # === 1. Фізичні підкроки ===
        for _ in range(self.substeps):
            SIM_FIELD.detect_collision()
            
            for body in SIM_BODIES:
                body.validate_joints()
                
            for body in SIM_BODIES:
                 body.joints = [j for j in body.joints if j.active and (j.Body_1 and j.Body_2)]
            
            # Паралельний розрахунок
            if self._executor:
                 list(self._executor.map(self.compute_body_motion, SIM_BODIES))
            else:
                for body in SIM_BODIES:
                    self.compute_body_motion(body)

        # === 2. Візуальна синхронізація (тільки для очей) ===
        for body in SIM_BODIES:
            body.sync_to_blender()

        # === 3. ЗАПИС У ПАМ'ЯТЬ (БЕЗ BLENDER API) ===
        self._current_rec_frame += 1.0
        for body in SIM_BODIES:
            body.record_frame(self._current_rec_frame)

    def compute_body_motion(self, body):
        body.slerp_motion(self.dt_sub)

    def execute(self, context):
        if self.__class__._is_running: return {'CANCELLED'}
        self.__class__._is_running = True
        
        # Починаємо запис з поточного кадру
        self._current_rec_frame = float(context.scene.frame_current)
        
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)
        wm = context.window_manager
        self._timer = wm.event_timer_add(self.dt_frame, window=context.window)
        wm.modal_handler_add(self)
        
        print(">>> Physics STARTED. Press ESC to Stop & Bake. <<<")
        return {'RUNNING_MODAL'}

    def cancel(self, context):
        self.__class__._is_running = False
        
        wm = context.window_manager
        if self._timer:
            wm.event_timer_remove(self._timer)
        if self._executor:
            self._executor.shutdown(wait=False)
            self._executor = None
            
        # === МОМЕНТ ІСТИНИ: ЗАПІКАННЯ ===
        print(">>> Physics Stopped. BAKING KEYFRAMES... <<<")
        self.bake_simulation(context)
        
        return {'CANCELLED'}

    def bake_simulation(self, context):
        """Переносить дані з пам'яті Python у ключі Blender"""
        global SIM_BODIES
        
        # Змінюємо курсор, щоб показати, що йде робота
        wm = context.window_manager
        wm.progress_begin(0, 100)
        
        # Вимикаємо анімацію перед записом, щоб прискорити процес
        # (Ось тут ми можемо безпечно очистити стару анімацію, бо нова вже в пам'яті)
        for body in SIM_BODIES:
            print ('joints',len(body.joints), body.dimensions,body.linked_object.name)
            if not body.is_static and body.recorder:
                 # Якщо хочете зберегти стару анімацію ДО симуляції, 
                 # цей рядок треба прибрати або зробити розумнішим.
                 # Але для чистого результату краще очистити:
                 body.linked_object.animation_data_clear() 
        
        total_bodies = len(SIM_BODIES)
        
        for i, body in enumerate(SIM_BODIES):
            if body.is_static or not body.recorder:
                continue
                
            obj = body.linked_object
            
            # Створюємо Action, якщо його немає
            if not obj.animation_data:
                obj.animation_data_create()
            if not obj.animation_data.action:
                obj.animation_data.action = bpy.data.actions.new(name=f"{obj.name}_Physics")
            
            # --- Швидкий спосіб вставки ключів ---
            # Замість keyframe_insert на кожному кадрі, ми просто йдемо по записаному списку
            
            # 1. Проходимо по записаній історії
            for frame, matrix in body.recorder:
                # Встановлюємо матрицю
                obj.matrix_world = matrix
                
                # Вставляємо ключі (це все ще повільно, але надійніше)
                # Ми явно вказуємо options={'INSERTKEY_NEEDED'}, щоб не писати зайвого
                obj.keyframe_insert(data_path="location", frame=frame)
                obj.keyframe_insert(data_path="rotation_quaternion", frame=frame)
            
            # Оновлюємо прогрес бар
            wm.progress_update(int((i / total_bodies) * 100))

        wm.progress_end()
        print(">>> BAKING FINISHED <<<")
    
classes = (PhysicsSetupOperator,PhysicsModalOperator)
def register():
    for cls in classes:
        register_class(cls)

def unregister():
    from bpy.utils import unregister_class
    for cls in reversed(classes):
        unregister_class(cls)
if __name__ == "__main__":
    register()
