from copy import copy
import bpy
import bgl
import blf
import time
import concurrent.futures
from bpy.props import (StringProperty,
                       BoolProperty,
                       PointerProperty,
                       IntProperty,
                       FloatProperty,
                       CollectionProperty
                       )
from bpy.types import (Panel,
                       Menu,
                       Operator,
                       PropertyGroup,
                       )
from bpy.utils import register_class

class OBJECT_PT_TestSpringPhE(Panel):
    bl_label = "Spring Physics"
    bl_idname = "OBJECT_PT_TestSpringPhE"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "SpringPhysics"
    bl_context = "objectmode"


    @classmethod
    def poll(self,context):
        return context.object is not None

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        mytool = scene.render
        l0 = layout.operator("physics.setup_and_run", text="Init Simulation", icon='MESH_PLANE')
        l1 = layout.operator("wm.physics_modal_operator", text="Run Simulation", icon='MESH_PLANE')
classes = (OBJECT_PT_TestSpringPhE,)
def register():
    for cls in classes:
        register_class(cls)

def unregister():
    from bpy.utils import unregister_class
    for cls in reversed(classes):
        unregister_class(cls)
    #del bpy.types.Scene.my_tool


if __name__ == "__main__":
    register()
