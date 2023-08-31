from flask import (
    Blueprint, flash, g, redirect, render_template, request, url_for
)
from werkzeug.exceptions import abort
from werkzeug.utils import secure_filename

from flaskprocaudio.auth import login_required
from flaskprocaudio.db import get_db
import os
import shutil

from pathlib import Path
this_dir = Path(globals().get("__file__", "./_")).absolute().parent

from threading import Thread, Lock
from .audioprocessing import rundebruitage

ALLOWED_EXTENSIONS = {'wav'}
UPLOAD_FOLDER = os.path.join(this_dir, 'audio', 'uploaded')
# max num of uploaded files by user
MAX_INPUT_TRACKS = 3

bp = Blueprint('blog', __name__)


def allowed_file(filename):
    return '.' in filename and filename.split('.')[-1].lower() in ALLOWED_EXTENSIONS


@bp.route('/upload', methods=('GET', 'POST'))
@login_required
def upload():
    if request.method == 'POST':
        # title = request.form['title']
        # body = request.form['body']
        error = None
        if 'file' not in request.files:
            error = 'No file part'
        
        
        else:
            file = request.files['file']
        
            if not file.filename:
                error = 'filename is required.'
            elif not allowed_file(file.filename):
                error = f'filename {file.filename} extension is not allowed'

            if error is not None:
                flash(error)        
            else:
                
                filename = secure_filename(file.filename)
                db = get_db()
                # get user count of tracks
                user_check = db.execute("SELECT COUNT(*) FROM post WHERE author_id = ?",
                                         (g.user['id'],)).fetchone()
                if user_check[0] >= MAX_INPUT_TRACKS:  # there is already a file for this user
                    # get user files
                    user_track_filename = db.execute("SELECT filename FROM post WHERE author_id = ?",
                                         (g.user['id'],)).fetchall()
                    print(len(user_track_filename))
                    for tr in user_track_filename:
                        print(tr['filename'])

                    flash('removing some uploaded files')
                    for track in user_track_filename[:(1+len(user_track_filename)-MAX_INPUT_TRACKS)]:
                        trackname = track['filename']
                        print(trackname)
                        # remove file physical
                        phystrackname = os.path.join(UPLOAD_FOLDER, trackname)
                        if os.path.isfile(phystrackname):
                            os.remove(phystrackname)
                        # remove from db
                        db.execute('DELETE FROM post WHERE filename = ? and author_id = ?', (trackname, g.user['id']))
                        db.commit()


                file.save(os.path.join(UPLOAD_FOLDER, filename))
                
                db.execute(
                    'INSERT INTO post (filename, author_id)'
                    ' VALUES (?, ?)',
                    (filename, g.user['id'])
                )
                db.commit()
                rundebruitage(os.path.join(UPLOAD_FOLDER, filename))
                return redirect(url_for('blog.index'))

    return render_template('blog/upload.html')


def get_post(id, check_author=True):
    post = get_db().execute(
        'SELECT p.id, filename, created, author_id, username'
        ' FROM post p JOIN user u ON p.author_id = u.id'
        ' WHERE p.id = ?',
        (id,)
    ).fetchone()

    if post is None:
        abort(404, f"Post id {id} doesn't exist.")

    if check_author and post['author_id'] != g.user['id']:
        abort(403)

    return post


@bp.route('/')
def index():
    db = get_db()
    posts = db.execute(
        'SELECT p.id, filename, created, author_id, username'
        ' FROM post p JOIN user u ON p.author_id = u.id'
        ' ORDER BY created DESC'
    ).fetchall()
    return render_template('blog/index.html', posts=posts)
        
"""        
@bp.route('/<int:id>/update', methods=('GET', 'POST'))
@login_required
def update(id):
    post = get_post(id)

    if request.method == 'POST':
        title = request.form['title']
        body = request.form['body']
        error = None

        if not title:
            error = 'Title is required.'

        if error is not None:
            flash(error)
        else:
            db = get_db()
            db.execute(
                'UPDATE post SET title = ?, body = ?'
                ' WHERE id = ?',
                (title, body, id)
            )
            db.commit()
            return redirect(url_for('blog.index'))

    return render_template('blog/update.html', post=post)
"""

@bp.route('/<int:id>/delete', methods=('POST',))
@login_required
def delete(id):
    get_post(id)
    db = get_db()
    db.execute('DELETE FROM post WHERE id = ?', (id,))
    db.commit()
    return redirect(url_for('blog.index'))
        