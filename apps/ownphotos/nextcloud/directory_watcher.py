from api.models import (Photo, Person, LongRunningJob)
from django_rq import job
import owncloud as nextcloud
import pathlib
import ipdb
import os
from ownphotos import settings
import os
import datetime
import hashlib
import pytz
import time

from tqdm import tqdm
from config import image_dirs

import api.util as util

import ipdb
from django_rq import job
import time
import numpy as np
import rq
from api.directory_watcher import is_new_image, handle_new_image
from api.image_similarity import build_image_similarity_index


def collect_photos(nc, path, photos):
    for x in nc.list(path):
        if x.path.lower().endswith('.jpg'):
            photos.append(x.path)
        elif x.is_dir():
            collect_photos(nc, x.path, photos)


# @job
def scan_photos(user):
    job_id = rq.get_current_job().id

    if LongRunningJob.objects.filter(job_id=job_id).exists():
        lrj = LongRunningJob.objects.get(job_id=job_id)
        lrj.started_at = datetime.datetime.now().replace(tzinfo=pytz.utc)
        lrj.save()
    else:
        lrj = LongRunningJob.objects.create(
            started_by=user,
            job_id=job_id,
            queued_at=datetime.datetime.now().replace(tzinfo=pytz.utc),
            started_at=datetime.datetime.now().replace(tzinfo=pytz.utc),
            job_type=LongRunningJob.JOB_SCAN_PHOTOS)
        lrj.save()

    nc = nextcloud.Client(user.nextcloud_server_address)
    nc.login(user.nextcloud_username, user.nextcloud_app_password)

    scan_directory = user.nextcloud_scan_directory
    photos = []

    image_paths = []

    collect_photos(nc, scan_directory, photos)

    for photo in photos:
        local_dir = os.path.join(settings.BASE_DIR, 'nextcloud_media',
                                 user.username,
                                 os.path.dirname(photo)[1:])
        local_path = os.path.join(settings.BASE_DIR, 'nextcloud_media',
                                  user.username, photo[1:])
        image_paths.append(local_path)

        if not os.path.exists(local_dir):
            pathlib.Path(local_dir).mkdir(parents=True, exist_ok=True)

        if not os.path.exists(local_path):
            nc.get_file(photo, local_path)
        util.logger.info('Downloaded photo from nextcloud to '+local_path)

    try:
        image_paths.sort()

        existing_hashes = [p.image_hash for p in Photo.objects.all()]

        image_paths_to_add = []
        for image_path in image_paths:
            if not Photo.objects.filter(image_path=image_path).exists():
                image_paths_to_add.append(image_path)

        added_photo_count = 0
        already_existing_photo = 0
        to_add_count = len(image_paths_to_add)
        for idx, image_path in enumerate(image_paths_to_add):
            util.logger.info('begin handling of photo %d/%d'%(idx+1,to_add_count))
            handle_new_image(user, image_path, job_id)
            lrj.result = {
                'progress': {
                    "current": idx + 1,
                    "target": to_add_count
                }
            }
            lrj.save()

        util.logger.info("Added {} photos".format(len(image_paths_to_add)))
        build_image_similarity_index(user)

        lrj = LongRunningJob.objects.get(job_id=rq.get_current_job().id)
        lrj.finished = True
        lrj.finished_at = datetime.datetime.now().replace(tzinfo=pytz.utc)
        prev_result = lrj.result
        next_result = prev_result
        next_result['new_photo_count'] = added_photo_count
        lrj.result = next_result
        lrj.save()
    except Exception as e:
        util.logger.error(str(e))
        util.logger.error(str(traceback.format_exc()))
        lrj = LongRunningJob.objects.get(job_id=rq.get_current_job().id)
        lrj.finished = True
        lrj.failed = True
        lrj.finished_at = datetime.datetime.now().replace(tzinfo=pytz.utc)
        prev_result = lrj.result
        next_result = prev_result
        next_result['new_photo_count'] = 0
        lrj.result = next_result
        lrj.save()
    return {"new_photo_count": added_photo_count, "status": True}
