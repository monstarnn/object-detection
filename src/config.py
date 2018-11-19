import argparse, jinja2, sys


def main():
    build_config()


def build_config():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir')
    parser.add_argument('--num_steps', default=50000)
    parser.add_argument('--resize_min_dimension', type=positive_int, default=600)
    parser.add_argument('--resize_max_dimension', type=positive_int, default=1024)
    parser.add_argument('--resize_fixed_width', type=positive_int, default=0)
    parser.add_argument('--resize_fixed_height', type=positive_int, default=0)
    args, _ = parser.parse_known_args()

    targs = {
        'data_dir': args.data_dir,
        'resize_min_dimension': args.resize_min_dimension,
        'resize_max_dimension': args.resize_max_dimension,
        'resize_fixed_width': args.resize_fixed_width,
        'resize_fixed_height': args.resize_fixed_height,
        'num_steps': args.num_steps,
    }

    if targs['resize_min_dimension'] == 0 or targs['resize_min_dimension'] == 0:
        targs['resize_min_dimension'] = 0
        targs['resize_max_dimension'] = 0
    if targs['resize_fixed_width'] == 0 or targs['resize_fixed_height'] == 0:
        targs['resize_fixed_width'] = 0
        targs['resize_fixed_height'] = 0

    if targs['resize_min_dimension'] == 0 and targs['resize_fixed_width'] == 0:
        raise Exception('it should be set positive resize_min_dimension with resize_max_dimension or resize_fixed_width with resize_fixed_height')

    # print(targs)
    # print(targs['resize_min_dimension'])

    t = open("faster_rcnn_resnet101_pets.config.template", "r")
    template = jinja2.Template(t.read())
    t.close()
    tw = open("faster_rcnn_resnet101_pets.config", "w+")
    cfg = str(template.render(args=targs))
    print(cfg)
    tw.write(cfg)
    tw.close()


def positive_int(val):
    ival = 0
    try:
        ival = int(val)
    except:
        pass
    finally:
        return ival if ival > 0 else 0


if __name__ == '__main__':
    main()
