import argparse, jinja2


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='$DATA_DIR')
    parser.add_argument('--num_steps', default=50000)

    args = parser.parse_args()
    t = open("faster_rcnn_resnet101_pets.config.template", "r")
    template = jinja2.Template(t.read())
    t.close()
    tw = open("faster_rcnn_resnet101_pets.config2", "w+")
    tw.write(str(template.render(args=args)))
    tw.close()


if __name__ == '__main__':
    main()
