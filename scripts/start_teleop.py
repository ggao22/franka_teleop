#!/home/george/miniforge3/envs/polymetis-local/bin/python
import rospy
import click

from start_teleop_node import FrankaTeleop

@click.command()
@click.option('--demo_output_file', '-o', required=False, default=None)
def main(demo_output_file):
    if demo_output_file:
        ft = FrankaTeleop(rate=10.0,
                          demo_output_file=demo_output_file)
        ft.start_demo_recorder()
    else:
        ft = FrankaTeleop()
        ft.start_teleop()
    

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
