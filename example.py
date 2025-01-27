
from ai2thor.controller import Controller
import time
import curses
import os,sys,termios,tty
import getch
def get_key():
    # fd = sys.stdin.fileno()
    # old_settings = termios.tcgetattr(fd)
    # try:
    #     tty.setraw(sys.stdin.fileno())
    #     ch = sys.stdin.read(1)
    # finally:
    #     termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return getch.getch()
def rotate_left(controller, degrees=90, steps=10, step_delay=0.05):
    step_size = degrees / steps

    controller.step("RotateLeft", degrees=30)
    time.sleep(step_delay)

def rotate_right(controller, degrees=90, steps=10, step_delay=0.05):
    step_size = degrees / steps

    controller.step("RotateRight", degrees=30)
    time.sleep(step_delay)

def move_ahead(controller, meters, step_delay=0.05):
    steps = int(meters * 4)
    # for _ in range(steps):
    controller.step("MoveAhead")
    time.sleep(step_delay)

#         if env_type == EnvTypes.ROBOTHOR:
#             self.controller_kwargs["commit_id"] = self.config["thor_build_id"]
#         elif env_type == EnvTypes.NORMAL:
#             self.controller_kwargs["local_executable_path"] = './pasture_builds/thor_build_normal/normal.x86_64'
#         elif env_type == EnvTypes.DUP:
#             self.controller_kwargs["local_executable_path"] = './pasture_builds/thor_build_dup/dup.x86_64'
#         elif env_type == EnvTypes.REMOVE:
#             self.controller_kwargs["local_executable_path"] = './pasture_builds/thor_build_remove/remove.x86_64'
#         elif env_type == EnvTypes.LONGTAIL:
#             self.controller_kwargs["local_executable_path"] = './pasture_builds/thor_build_longtail/longtail.x86_64'
#         else:
#             raise ValueError('unsupported env_type')

def teleport_action(controller,x,z):
    teleport_action = {
        "action": "Teleport",
        "position": {
            "x": x,
            "y": -7,
            "z": z
        },
        "rotation": {"x": 0, "y": 0, "z": 0},
        "horizon": 0,
    }
    print("teleport action")
    controller.step(action=teleport_action)

def main():

    controller = Controller(
        agentMode="default",
        visibilityDistance=1.0,
        gridSize=0.25,
        fieldOfView=90,
        local_executable_path = "./pasture_builds/thor_build_normal/normal.x86_64",
        width=672,
        height=672
    )

    controller.reset("FloorPlan_Val1_1")

    controller.step(
        action="LookDown",
        degrees=30
    )

    teleport_action = {
        "action": "Teleport",
        "position": {
            "x": 3,
            "y": -7,
            "z": -12
        },
        "rotation": {"x": 0, "y": 0, "z": 0},
        "horizon": 0,
    }
    print("Action : Teleport Action")
    controller.step(action=teleport_action)





    # commands = "2 R 11 R 8 R 11 R 9 L 7 L 3 R 10 L 1 L 5 R 7"

    # Split the command string into a single command
    # command_list = commands.split()

    while True:

        event = controller.last_event

        lastAction = event.metadata['lastActionSuccess']
        agent_info = event.metadata['agent']
        reachable_positions = controller.step(
            action="GetReachablePositions"
        ).metadata["actionReturn"]
        print(f"last action : {lastAction}")
        print("Agent Info = ", agent_info)
        print("Reachable Position = ", reachable_positions)

        key = input("Action :")
        if key == 'w':
            print("Action : MoveAhead")
            move_ahead(controller, 0)
        elif key == 'a':
            print("Action : Rotate Left")
            rotate_left(controller)
        elif key == 'd':
            print("Action : Rotate Right")
            rotate_right(controller)
        elif key == 't':
            try:
                new_coords = input("Enter new coordinates: ").split()
                agent_x = 0
                agent_y = 0
                print(new_coords)
                if len(new_coords) == 2:
                    agent_x = int(new_coords[0])
                    agent_y = int(new_coords[1])
                teleport_action(controller,agent_x,agent_y)
            except Exception as e:
                print(e)




        # command = "L"
        # rotate_left(controller)

    # for command in command_list:
    #     if command.isdigit():
    #         move_ahead(controller, int(command) * 0.75)
    #     elif command == 'R':
    #         rotate_right(controller)
    #     elif command == 'L':
    #         rotate_left(controller)


if __name__ == '__main__':
    main()