import bpy
import functools


def execute_current_script():
    text_area = next(area for area in bpy.context.screen.areas if area.type == 'TEXT_EDITOR')
    current_text = text_area.spaces.active.text
    if current_text:
        script = f"exec(next(area for area in bpy.context.screen.areas if area.type == 'TEXT_EDITOR').spaces.active.text.as_string())"
        if script:
            console_area = next(area for area in bpy.context.screen.areas if area.type == 'CONSOLE')
            console_space = next(space for space in console_area.spaces if space.type == 'CONSOLE')
            override = bpy.context.copy()
            override['area'] = console_area
            override['region'] = next(region for region in console_area.regions if region.type == 'WINDOW')
            with bpy.context.temp_override(**override):
                bpy.ops.console.clear_line()
                bpy.ops.console.insert(text=script)
                bpy.ops.console.execute()
        else:
            print("No script content to execute.")
    else:
        print("No active text block.")


# Register the key map
addon_keymaps = []


def register_keymap():
    wm = bpy.context.window_manager
    kc = wm.keyconfigs.addon
    if kc:
        km = kc.keymaps.new(name='Text', space_type='TEXT_EDITOR')
        kmi = km.keymap_items.new("wm.call_menu", "F5", "PRESS")
#        kmi = km.keymap_items.new("bpy.ops.text.execute_current_script", "F5", "PRESS")
        kmi.properties.name = "TEXT_MT_execute_current_script"

        addon_keymaps.append((km, kmi))

# Define the operator class


class ExecuteCurrentScriptOperator(bpy.types.Operator):
    bl_idname = "text.execute_current_script"
    bl_label = "Execute Current Script"

    @classmethod
    def poll(cls, context):
        return context.area.type == 'TEXT_EDITOR'

    def execute(self, context):
        execute_current_script()
        return {'FINISHED'}

# Define the menu class


class TEXT_MT_ExecuteCurrentScriptMenu(bpy.types.Menu):
    bl_label = "Execute Current Script"
    bl_idname = "TEXT_MT_execute_current_script"

    def draw(self, context):
        layout = self.layout
        layout.operator("text.execute_current_script")

# Register the classes and keymap


def register():
    bpy.utils.register_class(ExecuteCurrentScriptOperator)
    bpy.utils.register_class(TEXT_MT_ExecuteCurrentScriptMenu)
    register_keymap()


def unregister():
    bpy.utils.unregister_class(ExecuteCurrentScriptOperator)
    bpy.utils.unregister_class(TEXT_MT_ExecuteCurrentScriptMenu)
    for km, kmi in addon_keymaps:
        km.keymap_items.remove(kmi)
    addon_keymaps.clear()


if __name__ == "__main__":
    register()
    unregister()
    register()
