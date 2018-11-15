##
#  Runs train -> eval -> export.
##
import argparse
import logging
import re
import sys

# from jinja2 import Template

from mlboardclient.api import client


SUCCEEDED = 'Succeeded'

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
mlboard = client.Client()
run_tasks = ['train', 'eval', 'export']


def override_task_arguments(task, params):
    for k, v in params.items():
        pattern = re.compile('--{}[ =]([^\s]+|[\'"].*?[\'"])'.format(k))
        resource = task.config['resources'][0]
        task_cmd = resource['command']
        replacement = '--{} {}'.format(k, v)
        if pattern.findall(task_cmd):
            # Replace
            resource['command'] = pattern.sub(
                replacement,
                task_cmd
            )
        else:
            # Add
            if 'args' in resource:
                resource['args'][k] = v
            else:
                resource['args'] = {k: v}


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir')
    parser.add_argument('--num_steps')
    # parser.add_argument('--convert', type=boolean_string, default=False)
    # parser.add_argument('--push-model', type=boolean_string, default=False)

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    override_args = {
        'train': {
            'data_dir': args.data_dir,
            'num_steps': args.num_steps,
        },
        'eval': {
            'data_dir': args.data_dir,
        },
        'export': {
            'data_dir': args.data_dir,
        },
    }

    # if args.convert:
    #     # Convert model before use it
    #     run_tasks.insert(0, 'model-converter')
    #     override_args['model-converter'] = {
    #         'push_model': str(args.push_model)
    #     }
    # else:
    #     # Use converted model
    #     override_args['train-classifier']['model'] = '$FACENET_DIR/facenet.xml'
    #     override_args['validate-classifier']['model'] = '$FACENET_DIR/facenet.xml'

    # t = open("faster_rcnn_resnet101_pets.config.template", "r")
    # template = Template(t.read())
    # t.close()
    # tw = open("faster_rcnn_resnet101_pets.config", "w+")
    # tw.write(str(template.render(args=args)))
    # tw.close()

    app = mlboard.apps.get()

    faces_set = None
    for task in run_tasks:
        t = app.tasks.get(task)
        # if faces_set is not None:
        #     revs = t.config.get('datasetRevisions',[])
        #     _revs = []
        #     for r in revs:
        #         if r['volumeName'] != 'faces':
        #             _revs.append(r)
        #     _revs.append({'revision': faces_set,'volumeName': 'faces'})
        #     t.config['datasetRevisions'] = _revs
        if t.name in override_args and override_args[t.name]:
            override_task_arguments(t, override_args[t.name])

        LOG.info("Start task %s..." % t.name)
        started = t.start()

        LOG.info(
            "Run & wait [name=%s, build=%s, status=%s]"
            % (started.name, started.build, started.status)
        )
        completed = started.wait()

        if completed.status != SUCCEEDED:
            LOG.warning(
                "Task %s-%s completed with status %s."
                % (completed.name, completed.build, completed.status)
            )
            LOG.warning(
                'Please take a look at the corresponding task logs'
                ' for more information about failure.'
            )
            LOG.warning("Workflow completed with status ERROR")
            sys.exit(1)

        LOG.info(
            "Task %s-%s completed with status %s."
            % (completed.name, completed.build, completed.status)
        )
        # if task=='align-images':
        #     faces_set = completed.exec_info['push_version']
        #     LOG.info('Setup new faceset: {}'.format(faces_set))
    LOG.info("Workflow completed with status SUCCESS")


def boolean_string(s):
    s = s.lower()
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'


if __name__ == '__main__':
    main()