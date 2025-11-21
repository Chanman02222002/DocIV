# ============================================================
#                      IMPORTS (CLEANED)
# ============================================================

from flask import (
    Flask, render_template, redirect, url_for, request,
    flash, send_from_directory, jsonify, send_file
)

from flask_sqlalchemy import SQLAlchemy
from flask_wtf import FlaskForm

from wtforms import (
    StringField, SubmitField, SelectMultipleField, FloatField,
    BooleanField, FieldList, FormField, SelectField,
    DateTimeLocalField, TextAreaField
)
from wtforms.validators import DataRequired, Email, Optional
from wtforms.widgets import ListWidget, CheckboxInput

from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

from flask_login import (
    LoginManager, login_user, logout_user, login_required,
    UserMixin, current_user
)

from datetime import datetime
from collections import defaultdict
from jinja2 import DictLoader
from pathlib import Path
from geopy.geocoders import Nominatim
from dotenv import load_dotenv

import threading
import webbrowser
import json
import os
import re
import pandas as pd
import time

from io import BytesIO
import openai

# ============================================================
#                      FLASK APP SETUP
# ============================================================

print("Current working directory:", os.getcwd())
print("Database path:", os.path.abspath('crm.db'))

app = Flask(__name__, static_folder="static")
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///crm.db"

# Load SECRET_KEY from environment or fallback
load_dotenv()
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "fallback-secret")

# ============================================================
#                      DATABASE + LOGIN
# ============================================================

db = SQLAlchemy(app)

login_manager = LoginManager(app)
login_manager.login_view = "login"

# ============================================================
#                      USER MODEL
# ============================================================

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)

    username = db.Column(db.String(150), unique=True, nullable=False)
    email    = db.Column(db.String(100), unique=True, nullable=False)

    password_hash = db.Column(db.Text, nullable=False)
    role = db.Column(db.String(20), default='client')

    # One-to-one doctor profile
    doctor = db.relationship("Doctor", back_populates="user", uselist=False)

    # Utility
    def set_password(self, pw):
        self.password_hash = generate_password_hash(pw)

    def check_password(self, pw):
        return check_password_hash(self.password_hash, pw)


# ============================================================
#                      JOB MODEL
# ============================================================

class Job(db.Model):
    id = db.Column(db.Integer, primary_key=True)

    title       = db.Column(db.String(100))
    location    = db.Column(db.String(100))
    salary      = db.Column(db.String(50))
    description = db.Column(db.Text)

    poster_id   = db.Column(db.Integer, db.ForeignKey("user.id"))
    latitude    = db.Column(db.Float)
    longitude   = db.Column(db.Float)

    poster = db.relationship("User", backref="jobs")


# ============================================================
#                      MESSAGE MODEL
# ============================================================

class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)

    sender_id    = db.Column(db.Integer, db.ForeignKey("user.id"))
    recipient_id = db.Column(db.Integer, db.ForeignKey("user.id"))
    job_id       = db.Column(db.Integer, db.ForeignKey("job.id"))
    doctor_id    = db.Column(db.Integer, db.ForeignKey("doctor.id"))

    content      = db.Column(db.Text)
    timestamp    = db.Column(db.DateTime, default=datetime.utcnow)
    read         = db.Column(db.Boolean, default=False)

    # general, interest, invite, invite_response
    message_type = db.Column(db.String(50), default="general")

    sender    = db.relationship("User", foreign_keys=[sender_id], backref="sent_messages")
    recipient = db.relationship("User", foreign_keys=[recipient_id], backref="received_messages")

    job    = db.relationship("Job", backref="messages")
    doctor = db.relationship("Doctor", backref="messages")


# ============================================================
#                 DOCTOR MODEL (CLEAN AND ORDERED)
# ============================================================

class Doctor(db.Model):
    id      = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), unique=True, nullable=False)

    user = db.relationship("User", back_populates="doctor")

    # Basic info
    position      = db.Column(db.String(10),  nullable=False)
    specialty     = db.Column(db.String(100))
    subspecialty  = db.Column(db.String(100))
    first_name    = db.Column(db.String(50))
    last_name     = db.Column(db.String(50))
    email         = db.Column(db.String(100), unique=True)

    phone         = db.Column(db.String(20))
    alt_phone     = db.Column(db.String(20))
    city_of_residence = db.Column(db.String(100))

    # MD/DO fields
    medical_school           = db.Column(db.String(100))
    med_grad_month_year      = db.Column(db.String(20))
    residency                = db.Column(db.String(100))
    residency_grad_month_year = db.Column(db.String(20))

    fellowship                   = db.Column(db.Text)   # comma-separated list
    fellowship_grad_month_year   = db.Column(db.Text)   # comma-separated list

    # NP/PA fields
    bachelors               = db.Column(db.String(100))
    bachelors_grad_month_year = db.Column(db.String(20))
    msn                    = db.Column(db.String(100))
    msn_grad_month_year    = db.Column(db.String(20))
    dnp                    = db.Column(db.String(100))
    dnp_grad_month_year    = db.Column(db.String(20))
    additional_training    = db.Column(db.Text)

    sponsorship_needed = db.Column(db.Boolean)

    # Malpractice (stored as JSON)
    malpractice_cases = db.Column(db.Text)

    # Certification
    certification                 = db.Column(db.String(30))
    certification_specialty_area  = db.Column(db.String(100))

    clinically_active     = db.Column(db.String(30))
    last_clinically_active = db.Column(db.String(20))

    emr       = db.Column(db.String(100))
    languages = db.Column(db.String(200))

    states_licensed        = db.Column(db.Text)
    states_willing_to_work = db.Column(db.Text)

    salary_expectations = db.Column(db.Float)
    joined = db.Column(db.DateTime, default=datetime.utcnow)

    profile_picture = db.Column(db.String(255))


# ============================================================
#               SCHEDULED CALL MODEL (CLEANED)
# ============================================================

class ScheduledCall(db.Model):
    id = db.Column(db.Integer, primary_key=True)

    doctor_id       = db.Column(db.Integer, db.ForeignKey("doctor.id"))
    scheduled_by_id = db.Column(db.Integer, db.ForeignKey("user.id"))
    job_id          = db.Column(db.Integer, db.ForeignKey("job.id"))

    datetime             = db.Column(db.DateTime, nullable=False)
    reason               = db.Column(db.String(255))
    canceled             = db.Column(db.Boolean, default=False)
    reschedule_requested = db.Column(db.Boolean, default=False)
    reschedule_note      = db.Column(db.String(255))
    reschedule_datetime  = db.Column(db.DateTime)

    invite_status = db.Column(db.String(20), default="Pending")  
    # Allowed: Pending, Accepted, Declined

    doctor       = db.relationship("Doctor", backref=db.backref("scheduled_calls", lazy=True))
    scheduled_by = db.relationship("User", backref=db.backref("scheduled_calls_scheduled", lazy=True))
    job          = db.relationship("Job", backref="scheduled_calls")


# ============================================================
#                 LOGIN MANAGER LOADER
# ============================================================

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# ============================================================
#      GLOBAL CONTEXT PROCESSOR (keeps current_user in Jinja)
# ============================================================

@app.context_processor
def inject_user():
    return dict(current_user=current_user)


# ============================================================
#                    GEOCODER UTILITY
# ============================================================

def geocode_location(location_str):
    """Convert a human-readable location into lat/lng."""
    geolocator = Nominatim(user_agent="job-map-geocoder")
    loc = geolocator.geocode(location_str)
    if loc:
        return loc.latitude, loc.longitude
    return None, None


# US state list used by multiple forms
states = [
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL",
    "IN","IA","KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT",
    "NE","NV","NH","NJ","NM","NY","NC","ND","OH","OK","OR","PA","RI",
    "SC","SD","TN","TX","UT","VT","VA","WA","WV","WI","WY"
]
# ============================================================
#                      WTForms — CLEANED
# ============================================================

# ------------------------------
#  HELPER SUBFORM (Checkboxes)
# ------------------------------

class MultiCheckboxField(SelectMultipleField):
    widget = ListWidget(prefix_label=False)
    option_widget = CheckboxInput()


# ============================================================
#                  AUTH / LOGIN / REGISTER FORMS
# ============================================================

class LoginForm(FlaskForm):
    username = StringField("Username", validators=[DataRequired()])
    password = StringField("Password", validators=[DataRequired()])
    submit = SubmitField("Login")


class RegisterForm(FlaskForm):
    username = StringField("Username", validators=[DataRequired()])
    password = StringField("Password", validators=[DataRequired()])
    email    = StringField("Email", validators=[DataRequired(), Email()])
    submit   = SubmitField("Register")


# ============================================================
#                  DOCTOR REGISTRATION FORM (ADMIN)
# ============================================================

class DoctorRegistrationForm(FlaskForm):
    username = StringField("Choose a Username", validators=[DataRequired()])
    email    = StringField("Doctor’s Email", validators=[DataRequired(), Email()])
    password = StringField("Password", validators=[DataRequired()])
    submit   = SubmitField("Create Doctor Account")


# ============================================================
#                  DOCTOR FULL PROFILE FORM
# ============================================================

class DoctorForm(FlaskForm):
    # Basic identity
    position   = SelectField("Position", choices=[
        ("MD", "MD"), ("DO", "DO"),
        ("NP", "NP"), ("PA", "PA")
    ], validators=[DataRequired()])

    specialty     = StringField("Specialty")
    subspecialty  = StringField("Subspecialty")

    first_name = StringField("First Name",  validators=[DataRequired()])
    last_name  = StringField("Last Name",   validators=[DataRequired()])
    email      = StringField("Email",       validators=[DataRequired(), Email()])

    phone     = StringField("Phone")
    alt_phone = StringField("Alternate Phone")
    city_of_residence = StringField("City of Residence")

    # MD/DO
    medical_school          = StringField("Medical School")
    med_grad_month_year     = StringField("Graduation (MM/YYYY)")
    residency               = StringField("Residency")
    residency_grad_month_year = StringField("Residency Grad (MM/YYYY)")

    fellowship                 = TextAreaField("Fellowship(s) — comma separated")
    fellowship_grad_month_year = TextAreaField("Fellowship Grad (comma separated)")

    # NP/PA
    bachelors               = StringField("Bachelors")
    bachelors_grad_month_year = StringField("Grad (MM/YYYY)")
    msn                    = StringField("MSN")
    msn_grad_month_year    = StringField("MSN Grad")
    dnp                    = StringField("DNP")
    dnp_grad_month_year    = StringField("DNP Grad")
    additional_training    = TextAreaField("Additional Training")

    # Immigration
    sponsorship_needed = BooleanField("Sponsorship Needed?")

    # Malpractice
    malpractice_cases = TextAreaField("Malpractice Cases (JSON or comma-separated)")

    # Certifications
    certification = SelectField(
        "Certification",
        choices=[
            ("Not Certified", "Not Certified"),
            ("Board Eligible", "Board Eligible"),
            ("Board Certified", "Board Certified")
        ]
    )
    certification_specialty_area = StringField("Certification Specialty Area")

    # Clinical activity
    clinically_active      = SelectField(
        "Clinically Active?",
        choices=[
            ("Currently Active", "Currently Active"),
            ("Not Currently Active", "Not Currently Active")
        ]
    )
    last_clinically_active = StringField("Last Active Month/Year")

    # EMR + Language
    emr       = StringField("EMR Experience")
    languages = StringField("Languages (comma separated)")

    # Licenses
    states_licensed = MultiCheckboxField(
        "States Licensed",
        choices=[(s, s) for s in states]
    )
    states_willing_to_work = MultiCheckboxField(
        "States Willing to Work",
        choices=[(s, s) for s in states]
    )

    salary_expectations = FloatField("Salary Expectations")

    # Profile picture file upload
    profile_picture = StringField("Profile Picture Path (auto-set by upload)")
    submit = SubmitField("Save Profile")


# ============================================================
#                    JOB POSTING FORM
# ============================================================

class JobForm(FlaskForm):
    title       = StringField("Job Title", validators=[DataRequired()])
    location    = StringField("Location", validators=[DataRequired()])
    salary      = StringField("Salary")
    description = TextAreaField("Description", validators=[DataRequired()])
    submit      = SubmitField("Post Job")


# ============================================================
#                  JOB SEARCH FILTERS (DOCTOR)
# ============================================================

class JobSearchForm(FlaskForm):
    keyword  = StringField("Keyword")
    location = StringField("Location")
    submit   = SubmitField("Search")


# ============================================================
#                       SCHEDULE CALL FORM
# ============================================================

class ScheduleCallForm(FlaskForm):
    doctor_id = SelectField("Select Doctor", coerce=int, validators=[DataRequired()])
    job_id    = SelectField("Related Job", coerce=int, validators=[Optional()])
    datetime  = DateTimeLocalField("Select Date/Time", validators=[DataRequired()])
    reason    = StringField("Reason for Call")
    send_invite = SelectField(
        "Send Invite to Doctor?",
        choices=[("yes", "Yes"), ("no", "No")],
        default="yes"
    )
    submit = SubmitField("Schedule Call")


# ============================================================
#               RESCHEDULE CALL REQUEST FORM (DOCTOR)
# ============================================================

class RescheduleRequestForm(FlaskForm):
    note      = StringField("Reason for Reschedule", validators=[DataRequired()])
    datetime  = DateTimeLocalField("New Proposed Time", validators=[DataRequired()])
    submit    = SubmitField("Submit Request")


# ============================================================
#          CLIENT RESPONSE FORM TO RESCHEDULE REQUEST
# ============================================================

class ClientRescheduleDecisionForm(FlaskForm):
    action = SelectField(
        "Decision",
        choices=[
            ("approve", "Approve Reschedule"),
            ("deny", "Deny Reschedule")
        ],
        validators=[DataRequired()]
    )
    submit = SubmitField("Submit Decision")


# ============================================================
#                   GENERAL MESSAGE FORM
# ============================================================

class MessageForm(FlaskForm):
    content = TextAreaField("Message", validators=[DataRequired()])
    submit  = SubmitField("Send Message")
# ============================================================
#                AUTHENTICATION & ACCOUNT ROUTES
# ============================================================

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    form = LoginForm()

    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data.strip()).first()

        if not user or not check_password_hash(user.password_hash, form.password.data):
            flash("Invalid username or password.", "danger")
            return redirect(url_for('login'))

        login_user(user)
        flash("Logged in successfully!", "success")
        return redirect(url_for('dashboard'))

    return render_template('login.html', form=form)
    
@app.route("/doctors")
@login_required
def doctors():
    all_doctors = Doctor.query.all()
    return render_template("doctors.html", doctors=all_doctors)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash("Logged out successfully.", "success")
    return redirect(url_for('login'))


# ---------------------------
#     ACCOUNT CREATION
# ---------------------------

@app.route('/register', methods=['GET', 'POST'])
@login_required         # Admin-only registration
def register():
    if current_user.role != 'admin':
        flash("Unauthorized access", "danger")
        return redirect(url_for('dashboard'))

    form = RegisterForm()

    if form.validate_on_submit():
        # Username unique?
        if User.query.filter_by(username=form.username.data).first():
            flash("Username already exists.", "danger")
            return redirect(url_for('register'))

        # Email unique?
        if User.query.filter_by(email=form.email.data).first():
            flash("Email already exists.", "danger")
            return redirect(url_for('register'))

        user = User(
            username=form.username.data.strip(),
            email=form.email.data.strip(),
            password_hash=generate_password_hash(form.password.data),
            role="user"
        )

        db.session.add(user)
        db.session.commit()

        flash("User registered successfully!", "success")
        return redirect(url_for('dashboard'))

    return render_template('register.html', form=form)


# =================================================================
#        PUBLIC REGISTRATION (CLIENT / DOCTOR SELF-SIGNUP)
# =================================================================

@app.route('/public_register_client', methods=['POST'])
def public_register_client():
    data = request.form
    username = data.get("username", "").strip()
    email = data.get("email", "").strip()
    password = data.get("password")
    confirm  = data.get("confirm_password")

    if password != confirm:
        flash("Passwords do not match.", "danger")
        return redirect(url_for('create_account'))

    # Unique username or email
    if User.query.filter((User.username == username) | (User.email == email)).first():
        flash("Username or email already exists.", "danger")
        return redirect(url_for('create_account'))

    user = User(username=username, email=email, role="client")
    user.set_password(password)

    db.session.add(user)
    db.session.commit()

    login_user(user)
    flash("Client account created!", "success")
    return redirect(url_for('client_dashboard'))

@app.route('/admin/inbox')
@login_required
def admin_inbox():
    if current_user.role != "admin":
        flash("Unauthorized", "danger")
        return redirect(url_for("dashboard"))

    messages = Message.query.filter_by(
        recipient_id=current_user.id
    ).order_by(Message.timestamp.desc()).all()

    for msg in messages:
        if not msg.read:
            msg.read = True
    db.session.commit()

    return render_template("admin_inbox.html", messages=messages)

@app.route('/public_register_doctor', methods=['POST'])
def public_register_doctor():
    data = request.form

    username = data.get("username", "").strip()
    email = data.get("email", "").strip()
    password = data.get("password")
    confirm  = data.get("confirm_password")
    first    = data.get("first_name", "")
    last     = data.get("last_name", "")
    specialty = data.get("specialty", "")

    if password != confirm:
        flash("Passwords do not match.", "danger")
        return redirect(url_for('create_account'))

    if User.query.filter((User.username == username) | (User.email == email)).first():
        flash("Username or email already exists.", "danger")
        return redirect(url_for('create_account'))

    # Create user
    user = User(username=username, email=email, role="doctor")
    user.set_password(password)
    db.session.add(user)
    db.session.commit()

    # Doctor profile
    doctor = Doctor(
        first_name=first,
        last_name=last,
        email=email,
        specialty=specialty,
        position="MD",
        user_id=user.id,
        joined=datetime.utcnow()
    )
    db.session.add(doctor)
    db.session.commit()

    login_user(user)
    flash("Doctor profile created!", "success")
    return redirect(url_for('doctor_dashboard'))


# ============================================================
#                    LANDING & HOME ROUTES
# ============================================================

@app.route('/')
def landing_page():
    return render_template("landing_page.html", current_year=datetime.now().year)


@app.route('/home')
def home():
    return redirect(url_for('dashboard'))


# ============================================================
#                  DASHBOARD REDIRECTION LOGIC
# ============================================================

@app.route('/dashboard')
@login_required
def dashboard():
    role = current_user.role

    if role == "admin":
        return redirect(url_for('admin_dashboard'))
    elif role == "client":
        return redirect(url_for('client_dashboard'))
    elif role == "doctor":
        return redirect(url_for('doctor_dashboard'))
    else:
        flash("Role not recognized.", "danger")
        return redirect(url_for('logout'))
# ============================================================
#                  DOCTOR DASHBOARD + EVENTS
# ============================================================

@app.route('/doctor/dashboard')
@login_required
def doctor_dashboard():
    if current_user.role != 'doctor':
        flash("Unauthorized", "danger")
        return redirect(url_for('dashboard'))

    doctor = current_user.doctor

    scheduled_calls = ScheduledCall.query.filter_by(doctor_id=doctor.id).all()
    pending_invites = ScheduledCall.query.filter_by(
        doctor_id=doctor.id, invite_status="Pending"
    ).all()

    events = []
    for call in scheduled_calls:
        # Determine calendar color/status
        if call.canceled:
            color, status = "#ff4d4d", "Canceled"
        elif call.reschedule_requested:
            color, status = "#3788d8", "Reschedule Requested"
        elif call.invite_status.lower() == "pending":
            color, status = "#ffc107", "Pending Invite"
        elif call.invite_status.lower() == "accepted":
            color, status = "#28a745", "Accepted"
        else:
            color, status = "#6c757d", "Scheduled"

        events.append({
            "id": call.id,
            "title": f"Call with {call.scheduled_by.username}",
            "start": call.datetime.isoformat(),
            "color": color,
            "status": status
        })

    return render_template(
        "doctor_dashboard.html",
        doctor=doctor,
        events=events,
        pending_invites=pending_invites
    )


# ============================================================
#                  DOCTOR JOB SEARCH / JOB BOARD
# ============================================================

@app.route('/doctor/jobs')
@login_required
def doctor_jobs():
    if current_user.role != "doctor":
        flash("Unauthorized", "danger")
        return redirect(url_for('dashboard'))

    keyword  = request.args.get('keyword', '').lower()
    location = request.args.get('location', '').lower()

    jobs_query = Job.query
    if keyword:
        jobs_query = jobs_query.filter(
            Job.title.ilike(f"%{keyword}%") |
            Job.description.ilike(f"%{keyword}%")
        )
    if location:
        jobs_query = jobs_query.filter(Job.location.ilike(f"%{location}%"))

    jobs = jobs_query.order_by(Job.id.desc()).all()

    # Group markers by coordinate (avoids overlapping pins)
    marker_groups = defaultdict(list)
    for job in jobs:
        if job.latitude and job.longitude:
            key = (round(job.latitude, 5), round(job.longitude, 5))
            marker_groups[key].append(job)

    job_markers = []
    for (lat, lng), grouped_jobs in marker_groups.items():
        job_markers.append({
            "lat": lat,
            "lng": lng,
            "jobs": [
                {"id": j.id, "title": j.title, "location": j.location, "salary": j.salary}
                for j in grouped_jobs
            ]
        })

    return render_template(
        "doctor_jobs.html",
        jobs=jobs,
        job_markers=job_markers,
        keyword=keyword,
        location=location
    )


# ============================================================
#               DOCTOR VIEW JOB + EXPRESS INTEREST
# ============================================================

@app.route('/doctor/job/<int:job_id>', methods=['GET', 'POST'])
@login_required
def view_job(job_id):
    if current_user.role != 'doctor':
        flash("Unauthorized", "danger")
        return redirect(url_for('dashboard'))

    job = Job.query.get_or_404(job_id)

    already_interested = Message.query.filter_by(
        sender_id=current_user.id,
        job_id=job.id,
        message_type="interest"
    ).first()

    if request.method == "POST":
        if already_interested:
            flash("You already expressed interest.", "info")
        else:
            message = Message(
                sender_id=current_user.id,
                recipient_id=job.poster_id,
                job_id=job.id,
                doctor_id=current_user.doctor.id,
                content=(
                    f"Dr. {current_user.doctor.first_name} "
                    f"{current_user.doctor.last_name} expressed interest in "
                    f"your job '{job.title}'."
                ),
                message_type="interest"
            )
            db.session.add(message)
            db.session.commit()
            flash("Interest sent!", "success")

        return redirect(url_for('doctor_jobs'))

    return render_template("view_job.html", job=job, already_interested=already_interested)


@app.route('/doctor/job/<int:job_id>/express_interest', methods=['POST'])
@login_required
def express_interest(job_id):
    if current_user.role != 'doctor':
        flash("Unauthorized", "danger")
        return redirect(url_for('doctor_jobs'))

    job = Job.query.get_or_404(job_id)

    msg = Message(
        sender_id=current_user.id,
        recipient_id=job.poster_id,
        job_id=job.id,
        doctor_id=current_user.doctor.id,
        content=(
            f"{current_user.doctor.first_name} {current_user.doctor.last_name} "
            f"expressed interest in your job: '{job.title}'."
        ),
        message_type="interest"
    )

    db.session.add(msg)
    db.session.commit()

    flash("Interest sent!", "success")
    return redirect(url_for('doctor_jobs'))


# ============================================================
#               DOCTOR AI JOB SEARCH (GPT-POWERED)
# ============================================================

@app.route('/doctor/ai_search_jobs', methods=['POST'])
@login_required
def doctor_ai_search_jobs():
    if current_user.role != "doctor":
        return jsonify({"error": "Unauthorized"}), 401

    lifestyle = request.form.get('lifestyle', '')
    wants     = request.form.get('wants', '')
    location  = request.form.get('location', '')

    jobs = Job.query.order_by(Job.id.desc()).all()

    jobs_payload = [{
        "id": job.id,
        "title": job.title,
        "location": job.location,
        "salary": job.salary,
        "description": job.description
    } for job in jobs]

    # GPT Prompt
    prompt = f"""
You are a professional assistant matching doctors with jobs.
(… full prompt unchanged …)
Job list:
{json.dumps(jobs_payload, indent=2)}
"""

    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            {"role": "system", "content": "You are a medical job match assistant."},
            {"role": "user",   "content": prompt}
        ],
        max_tokens=2200,
        temperature=0.5,
    )

    html = response.choices[0].message.content
    return jsonify({"html": html})


# ============================================================
#              DOCTOR CALL DETAILS (CANCEL / RESCHEDULE)
# ============================================================

@app.route('/doctor/call/<int:call_id>', methods=['GET', 'POST'])
@login_required
def doctor_call_details(call_id):
    if current_user.role != "doctor":
        flash("Unauthorized", "danger")
        return redirect(url_for('doctor_dashboard'))

    call = ScheduledCall.query.get_or_404(call_id)

    if call.doctor_id != current_user.doctor.id:
        flash("Unauthorized", "danger")
        return redirect(url_for('doctor_dashboard'))

    if request.method == "POST":
        action = request.form.get('action')

        if action == "cancel":
            call.canceled = True
            db.session.commit()
            flash("Meeting canceled.", "success")
            return redirect(url_for('doctor_dashboard'))

        elif action == "reschedule":
            new_dt = request.form.get('reschedule_datetime')
            note   = request.form.get('reschedule_note')

            call.reschedule_requested = True
            call.reschedule_note = note
            call.reschedule_datetime = datetime.strptime(new_dt, "%Y-%m-%dT%H:%M")

            db.session.commit()

            # Send notification
            message = Message(
                sender_id=current_user.id,
                recipient_id=call.scheduled_by_id,
                content=(
                    f"Reschedule requested for call on "
                    f"{call.datetime.strftime('%Y-%m-%d %H:%M')} "
                    f"to {call.reschedule_datetime.strftime('%Y-%m-%d %H:%M')}. "
                    f"Reason: {note}"
                )
            )
            db.session.add(message)
            db.session.commit()

            flash("Reschedule request sent.", "success")
            return redirect(url_for('doctor_dashboard'))

    return render_template("doctor_call_details.html", call=call)





# ============================================================
#                DOCTOR PUBLIC PROFILE (VIEW ONLY)
# ============================================================

@app.route('/doctor/<int:doctor_id>')
def doctor_profile(doctor_id):
    doctor = Doctor.query.get_or_404(doctor_id)
    malpractice_cases = json.loads(doctor.malpractice_cases or "[]")
    return render_template("doctor_profile.html", doctor=doctor, malpractice_cases=malpractice_cases)
# ============================================================
#                    CLIENT DASHBOARD + EVENTS
# ============================================================

@app.route('/client/dashboard')
@login_required
def client_dashboard():
    if current_user.role != "client":
        flash("Unauthorized", "danger")
        return redirect(url_for('dashboard'))

    scheduled_calls = ScheduledCall.query.filter_by(
        scheduled_by_id=current_user.id
    ).all()

    reschedule_requests = ScheduledCall.query.filter_by(
        scheduled_by_id=current_user.id,
        reschedule_requested=True
    ).all()

    events = []
    for call in scheduled_calls:
        if call.canceled:
            color, status = "#dc3545", "Canceled"
        elif call.reschedule_requested:
            color, status = "#17a2b8", "Reschedule Requested"
        elif call.invite_status == "Pending":
            color, status = "#ffc107", "Pending Invite"
        else:
            color, status = "#28a745", "Accepted"

        events.append({
            "id": call.id,
            "title": f"Call with Dr. {call.doctor.first_name} {call.doctor.last_name}",
            "start": call.datetime.isoformat(),
            "color": color,
            "status": status,
        })

    return render_template(
        "client_dashboard.html",
        events=events,
        reschedule_requests=reschedule_requests
    )


# ============================================================
#                        CLIENT INBOX
# ============================================================

@app.route('/client/inbox')
@login_required
def client_inbox():
    if current_user.role != "client":
        flash("Unauthorized", "danger")
        return redirect(url_for('dashboard'))

    messages = Message.query.filter_by(
        recipient_id=current_user.id
    ).order_by(Message.timestamp.desc()).all()

    # Mark unread messages as read
    for msg in messages:
        if not msg.read:
            msg.read = True
    db.session.commit()

    return render_template("inbox.html", messages=messages, title="Client Inbox")


# ============================================================
#                     CLIENT JOB POSTING
# ============================================================

@app.route('/post_job', methods=['GET', 'POST'])
@login_required
def post_job():
    if current_user.role not in ["client", "admin"]:
        flash("Unauthorized", "danger")
        return redirect(url_for('dashboard'))

    form = JobForm()

    if form.validate_on_submit():
        lat, lng = geocode_location(form.location.data)

        job = Job(
            title=form.title.data,
            location=form.location.data,
            salary=form.salary.data,
            description=form.description.data,
            poster_id=current_user.id,
            latitude=lat,
            longitude=lng
        )

        db.session.add(job)
        db.session.commit()

        flash("Job posted successfully!", "success")

        if current_user.role == "admin":
            return redirect(url_for('admin_dashboard'))
        return redirect(url_for('client_dashboard'))

    return render_template("post_job.html", form=form)


# ============================================================
#                         EDIT JOB
# ============================================================

@app.route('/edit_job/<int:job_id>', methods=['GET', 'POST'])
@login_required
def edit_job(job_id):
    job = Job.query.get_or_404(job_id)

    if current_user.role != "client" or job.poster_id != current_user.id:
        flash("Unauthorized", "danger")
        return redirect(url_for('dashboard'))

    form = JobForm(obj=job)

    if form.validate_on_submit():
        job.title       = form.title.data
        job.location    = form.location.data
        job.salary      = form.salary.data
        job.description = form.description.data

        db.session.commit()
        flash("Job updated successfully!", "success")
        return redirect(url_for('client_my_jobs'))

    return render_template("edit_job.html", form=form, job=job)


# ============================================================
#                   CLIENT VIEW: MY JOBS LIST
# ============================================================

@app.route('/client/my_jobs')
@login_required
def client_my_jobs():
    if current_user.role != "client":
        flash("Unauthorized", "danger")
        return redirect(url_for('dashboard'))

    keyword  = request.args.get('keyword', '').lower()
    location = request.args.get('location', '').lower()

    jobs_query = Job.query.filter_by(poster_id=current_user.id)

    if keyword:
        jobs_query = jobs_query.filter(
            Job.title.ilike(f"%{keyword}%") |
            Job.description.ilike(f"%{keyword}%")
        )
    if location:
        jobs_query = jobs_query.filter(Job.location.ilike(f"%{location}%"))

    jobs = jobs_query.order_by(Job.id.desc()).all()

    return render_template(
        "client_my_jobs.html",
        jobs=jobs,
        keyword=keyword,
        location=location
    )


# ============================================================
#               SEND INVITE TO DOCTOR (SCHEDULE CALL)
# ============================================================

@app.route('/send_invite/<int:doctor_id>/<int:job_id>', methods=['GET', 'POST'])
@login_required
def send_invite(doctor_id, job_id):
    if current_user.role not in ["client", "admin"]:
        flash("Unauthorized", "danger")
        return redirect(url_for('dashboard'))

    doctor = Doctor.query.get_or_404(doctor_id)
    doctor_user = User.query.get_or_404(doctor.user_id)
    job = Job.query.get_or_404(job_id)

    form = ScheduleCallForm()
    form.doctor_id.choices = [(doctor.id, f"{doctor.first_name} {doctor.last_name} | {doctor.email}")]

    if form.validate_on_submit():
        call = ScheduledCall(
            doctor_id=doctor.id,
            scheduled_by_id=current_user.id,
            job_id=job.id,
            datetime=form.datetime.data,
            reason=form.reason.data,
            invite_status="Pending"
        )

        db.session.add(call)
        db.session.commit()

        # Notify doctor
        message = Message(
            sender_id=current_user.id,
            recipient_id=doctor_user.id,
            job_id=job_id,
            doctor_id=doctor.id,
            content=f"You have a call invite from {current_user.username} on {form.datetime.data}.",
            message_type="invite"
        )
        db.session.add(message)
        db.session.commit()

        flash("Invite sent!", "success")

        if current_user.role == "admin":
            return redirect(url_for('admin_dashboard'))
        return redirect(url_for('client_dashboard'))

    return render_template("schedule_call.html", form=form, job=job, doctor=doctor)


# ============================================================
#                   CLIENT HANDLE RESCHEDULE REQUEST
# ============================================================

@app.route('/client/handle_reschedule/<int:call_id>', methods=['POST'])
@login_required
def client_handle_reschedule(call_id):
    """Client decides whether to approve or deny a doctor's reschedule request."""
    if current_user.role != "client":
        flash("Unauthorized", "danger")
        return redirect(url_for('dashboard'))

    call = ScheduledCall.query.get_or_404(call_id)
    action = request.form.get("action")
    client_note = request.form.get("client_note", "")

    if action == "accept":
        call.datetime = call.reschedule_datetime
        call.reschedule_requested = False
        call.reschedule_note = None
        call.reschedule_datetime = None

        content = (
            f"Your reschedule request for "
            f"{call.datetime.strftime('%Y-%m-%d %H:%M')} was approved. {client_note}"
        )

    elif action == "decline":
        declined_dt = call.reschedule_datetime
        call.reschedule_requested = False
        call.reschedule_note = None
        call.reschedule_datetime = None

        content = (
            f"Your reschedule request for "
            f"{declined_dt.strftime('%Y-%m-%d %H:%M')} was declined. {client_note}"
        )

    else:
        flash("Invalid action.", "danger")
        return redirect(url_for('client_dashboard'))

    db.session.commit()

    # Send message to doctor
    msg = Message(
        sender_id=current_user.id,
        recipient_id=call.doctor.user_id,
        content=content,
        timestamp=datetime.utcnow(),
        message_type="reschedule_result"
    )

    db.session.add(msg)
    db.session.commit()

    flash("Reschedule decision sent.", "success")
    return redirect(url_for('client_dashboard'))


# ============================================================
#     CLIENT → DOCTOR: SEND INVITE (ALREADY IN CLIENT SECTION)
# ============================================================

# This one was rewritten cleanly in Part 5:
# /send_invite/<doctor_id>/<job_id>


# ============================================================
#       STATUS COLORS FOR DASHBOARDS (USED EVERYWHERE)
# ============================================================

def get_call_status(call):
    """Shared helper to determine a call's display status."""
    if call.canceled:
        return "Canceled", "#dc3545"
    if call.reschedule_requested:
        return "Reschedule Requested", "#17a2b8"
    if call.invite_status == "Pending":
        return "Pending Invite", "#ffc107"
    if call.invite_status == "Accepted":
        return "Accepted", "#28a745"
    return "Scheduled", "#6c757d"
# ============================================================
#                     MESSAGING SYSTEM (CORE)
# ============================================================
# Includes:
# - Sending direct messages
# - Storing doctor ↔ client interest messages
# - Notification messages for invites/reschedules
# - Inbox rendering (client inbox exists in Part 5)
# - Mark-as-read handling
# ============================================================


# ============================================================
#                     SEND A GENERAL MESSAGE
# ============================================================

@app.route('/message/send/<int:recipient_id>', methods=['GET', 'POST'])
@login_required
def send_message(recipient_id):
    """Send a general message between users."""
    recipient = User.query.get_or_404(recipient_id)
    form = MessageForm()

    if form.validate_on_submit():
        msg = Message(
            sender_id=current_user.id,
            recipient_id=recipient.id,
            content=form.content.data,
            timestamp=datetime.utcnow(),
            message_type="general"
        )
        db.session.add(msg)
        db.session.commit()

        flash("Message sent!", "success")
        return redirect(url_for('view_conversation', user_id=recipient.id))

    return render_template("send_message.html", form=form, recipient=recipient)


# ============================================================
#                  VIEW CONVERSATION (THREAD)
# ============================================================

@app.route('/messages/<int:user_id>')
@login_required
def view_conversation(user_id):
    """Displays a message thread between current user and another user."""
    other = User.query.get_or_404(user_id)

    messages = Message.query.filter(
        ((Message.sender_id == current_user.id) & (Message.recipient_id == other.id)) |
        ((Message.sender_id == other.id) & (Message.recipient_id == current_user.id))
    ).order_by(Message.timestamp.asc()).all()

    # Mark unread messages as read
    for msg in messages:
        if msg.recipient_id == current_user.id and not msg.read:
            msg.read = True
    db.session.commit()

    return render_template("conversation.html", other=other, messages=messages)


# ============================================================
#                 DOCTOR INBOX (if you want one)
# ============================================================

@app.route('/doctor/inbox')
@login_required
def doctor_inbox():
    if current_user.role != "doctor":
        flash("Unauthorized", "danger")
        return redirect(url_for('dashboard'))

    messages = Message.query.filter_by(
        recipient_id=current_user.id
    ).order_by(Message.timestamp.desc()).all()

    # Mark as read
    for msg in messages:
        if not msg.read:
            msg.read = True
    db.session.commit()

    return render_template("inbox.html", messages=messages, title="Doctor Inbox")


# ============================================================
#     INTERNAL SYSTEM MESSAGES: INTEREST, INVITES, RESCHEDULES
# ============================================================

def send_system_message(sender_id, recipient_id, content, message_type):
    """Helper to send a standardized system message."""
    msg = Message(
        sender_id=sender_id,
        recipient_id=recipient_id,
        content=content,
        message_type=message_type,
        timestamp=datetime.utcnow()
    )
    db.session.add(msg)
    db.session.commit()
    return msg


# ============================================================
#      DOCTOR EXPRESSING INTEREST (BUSINESS LOGIC)
# ============================================================

def send_interest_message(job, doctor):
    """Used by both doctor interest routes."""
    content = (
        f"Dr. {doctor.first_name} {doctor.last_name} "
        f"expressed interest in your job '{job.title}'."
    )
    return send_system_message(
        sender_id=doctor.user_id,
        recipient_id=job.poster_id,
        content=content,
        message_type="interest"
    )


# ============================================================
#       SYSTEM MESSAGE TYPES USED THROUGHOUT THE APP
# ============================================================

# interest            — Doctor expresses job interest
# invite              — Client invites doctor to a call
# invite_response     — Doctor accepts/declines
# reschedule_request  — Doctor requests reschedule
# reschedule_result   — Client approves/denies
# general             — Freeform messaging between users


# ============================================================
#        CLEAN MESSAGE FETCHING UTILITY (DASHBOARD USE)
# ============================================================

def get_user_unread_messages(user_id):
    """Returns all unread messages for sidebar counters/etc."""
    return Message.query.filter_by(recipient_id=user_id, read=False).count()
# ============================================================
#                   IMAGE HELPERS
# ============================================================

def save_cropped_image(cropped_data):
    """Decode base64 cropped image and save to /static/upload."""
    import base64
    from PIL import Image
    from io import BytesIO

    if not cropped_data:
        return None

    try:
        header, encoded = cropped_data.split(",", 1)
        binary = base64.b64decode(encoded)
        img = Image.open(BytesIO(binary))

        filename = f"doctor_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.png"
        folder = os.path.join(app.static_folder, "upload")
        os.makedirs(folder, exist_ok=True)

        path = os.path.join(folder, filename)
        img.save(path)

        return f"upload/{filename}"
    except Exception:
        return None


# ============================================================
#                  DOCTOR EDIT PROFILE (SELF)
# ============================================================

@app.route('/doctor/edit_profile', methods=['GET', 'POST'])
@login_required
def doctor_edit_profile():
    if current_user.role != "doctor":
        flash("Unauthorized", "danger")
        return redirect(url_for('dashboard'))

    doctor = current_user.doctor
    form = DoctorForm()

    # -----------------------
    #        SUBMIT
    # -----------------------
    if form.validate_on_submit():

        # Prevent duplicate email
        check = Doctor.query.filter(
            Doctor.email == form.email.data,
            Doctor.id != doctor.id
        ).first()
        if check:
            flash("Another doctor already has this email.", "danger")
            return redirect(url_for('doctor_edit_profile'))

        # Handle cropped image
        cropped_data = request.form.get("cropped_image_data")
        if cropped_data:
            saved_path = save_cropped_image(cropped_data)
            if saved_path:
                doctor.profile_picture = saved_path

        # Simple fields
        doctor.position = form.position.data
        doctor.specialty = form.specialty.data
        doctor.subspecialty = form.subspecialty.data
        doctor.first_name = form.first_name.data
        doctor.last_name = form.last_name.data
        doctor.email = form.email.data
        doctor.phone = form.phone.data
        doctor.alt_phone = form.alt_phone.data
        doctor.city_of_residence = form.city_of_residence.data

        # MD/DO fields
        doctor.medical_school = form.medical_school.data
        doctor.med_grad_month_year = form.med_grad_month_year.data
        doctor.residency = form.residency.data
        doctor.residency_grad_month_year = form.residency_grad_month_year.data

        # Fellowships
        n_f = int(form.num_fellowships.data)
        doctor.fellowship = ",".join(form.fellowship.data[:n_f])
        doctor.fellowship_grad_month_year = ",".join(
            form.fellowship_grad_month_year.data[:n_f]
        )

        # NP/PA
        doctor.bachelors = form.bachelors.data
        doctor.bachelors_grad_month_year = form.bachelors_grad_month_year.data
        doctor.msn = form.msn.data
        doctor.msn_grad_month_year = form.msn_grad_month_year.data
        doctor.dnp = form.dnp.data
        doctor.dnp_grad_month_year = form.dnp_grad_month_year.data
        doctor.additional_training = form.additional_training.data
        doctor.sponsorship_needed = bool(form.sponsorship_needed.data)

        # Malpractice
        n_cases = int(form.num_malpractice_cases.data)
        cases = []
        for entry in form.malpractice_cases.entries[:n_cases]:
            cases.append({
                "incident_year": entry.form.incident_year.data,
                "outcome": entry.form.outcome.data,
                "payout_amount": entry.form.payout_amount.data or 0
            })
        doctor.malpractice_cases = json.dumps(cases)

        # Certification
        doctor.certification = form.certification.data
        doctor.certification_specialty_area = form.certification_specialty_area.data

        # Clinical activity
        doctor.clinically_active = form.clinically_active.data
        doctor.last_clinically_active = (
            form.last_clinically_active.data
            if form.clinically_active.data == "No"
            else None
        )

        # Misc fields
        doctor.emr = form.emr.data
        doctor.languages = form.languages.data

        # States
        doctor.states_licensed = ",".join(form.states_licensed.data)
        doctor.states_willing_to_work = ",".join(form.states_willing_to_work.data)

        doctor.salary_expectations = form.salary_expectations.data or 0.0

        db.session.commit()
        flash("Profile updated!", "success")
        return redirect(url_for('doctor_dashboard'))

    # -----------------------
    #        INITIAL POPULATE
    # -----------------------
    if request.method == 'GET':
        # Standard fields
        form.position.data = doctor.position
        form.specialty.data = doctor.specialty
        form.subspecialty.data = doctor.subspecialty
        form.first_name.data = doctor.first_name
        form.last_name.data = doctor.last_name
        form.email.data = doctor.email
        form.phone.data = doctor.phone
        form.alt_phone.data = doctor.alt_phone
        form.city_of_residence.data = doctor.city_of_residence

        form.medical_school.data = doctor.medical_school
        form.med_grad_month_year.data = doctor.med_grad_month_year
        form.residency.data = doctor.residency
        form.residency_grad_month_year.data = doctor.residency_grad_month_year

        # Fellowships
        f_list = doctor.fellowship.split(",") if doctor.fellowship else []
        f_dates = doctor.fellowship_grad_month_year.split(",") if doctor.fellowship_grad_month_year else []
        form.num_fellowships.data = str(len(f_list))

        form.fellowship.entries.clear()
        form.fellowship_grad_month_year.entries.clear()

        for f, d in zip(f_list, f_dates):
            fe = form.fellowship.append_entry()
            fe.data = f
            de = form.fellowship_grad_month_year.append_entry()
            de.data = d

        while len(form.fellowship.entries) < form.fellowship.max_entries:
            form.fellowship.append_entry()
            form.fellowship_grad_month_year.append_entry()

        # Malpractice
        m_list = json.loads(doctor.malpractice_cases or "[]")
        form.num_malpractice_cases.data = str(len(m_list))
        form.malpractice_cases.entries.clear()

        for c in m_list:
            entry = form.malpractice_cases.append_entry()
            entry.incident_year.data = c.get("incident_year", "")
            entry.outcome.data = c.get("outcome", "")
            entry.payout_amount.data = c.get("payout_amount", 0)

        while len(form.malpractice_cases.entries) < form.malpractice_cases.max_entries:
            form.malpractice_cases.append_entry()

        # Additional fields
        form.bachelors.data = doctor.bachelors
        form.bachelors_grad_month_year.data = doctor.bachelors_grad_month_year
        form.msn.data = doctor.msn
        form.msn_grad_month_year.data = doctor.msn_grad_month_year
        form.dnp.data = doctor.dnp
        form.dnp_grad_month_year.data = doctor.dnp_grad_month_year
        form.additional_training.data = doctor.additional_training
        form.sponsorship_needed.data = doctor.sponsorship_needed
        form.certification.data = doctor.certification
        form.certification_specialty_area.data = doctor.certification_specialty_area

        form.clinically_active.data = doctor.clinically_active
        form.last_clinically_active.data = doctor.last_clinically_active

        form.emr.data = doctor.emr
        form.languages.data = doctor.languages
        form.states_licensed.data = doctor.states_licensed.split(",") if doctor.states_licensed else []
        form.states_willing_to_work.data = doctor.states_willing_to_work.split(",") if doctor.states_willing_to_work else []
        form.salary_expectations.data = doctor.salary_expectations

    return render_template("doctor_edit_profile.html", form=form, doctor=doctor, zip=zip)


# ============================================================
#            ADMIN EDIT DOCTOR (EDIT_DOCTOR ROUTE)
# ============================================================

@app.route('/edit_doctor/<int:doctor_id>', methods=['GET', 'POST'])
@login_required
def edit_doctor(doctor_id):
    if current_user.role != "admin":
        flash("Unauthorized", "danger")
        return redirect(url_for('dashboard'))

    doctor = Doctor.query.get_or_404(doctor_id)
    form = DoctorForm()

    # ---------------------------------------
    # SAME LOGIC AS doctor_edit_profile()
    # ---------------------------------------
    # (This block is intentionally mirrored for admin use)

    if form.validate_on_submit():
        # Prevent duplicate email
        check = Doctor.query.filter(
            Doctor.email == form.email.data,
            Doctor.id != doctor.id
        ).first()
        if check:
            flash("Another doctor already has this email.", "danger")
            return redirect(url_for('edit_doctor', doctor_id=doctor.id))

        # Standard updates
        doctor.position = form.position.data
        doctor.specialty = form.specialty.data
        doctor.subspecialty = form.subspecialty.data
        doctor.first_name = form.first_name.data
        doctor.last_name = form.last_name.data
        doctor.email = form.email.data
        doctor.phone = form.phone.data
        doctor.alt_phone = form.alt_phone.data
        doctor.city_of_residence = form.city_of_residence.data

        doctor.medical_school = form.medical_school.data
        doctor.med_grad_month_year = form.med_grad_month_year.data
        doctor.residency = form.residency.data
        doctor.residency_grad_month_year = form.residency_grad_month_year.data

        # Fellowships
        n_f = int(form.num_fellowships.data)
        doctor.fellowship = ",".join(form.fellowship.data[:n_f])
        doctor.fellowship_grad_month_year = ",".join(
            form.fellowship_grad_month_year.data[:n_f]
        )

        # NP/PA
        doctor.bachelors = form.bachelors.data
        doctor.bachelors_grad_month_year = form.bachelors_grad_month_year.data
        doctor.msn = form.msn.data
        doctor.msn_grad_month_year = form.msn_grad_month_year.data
        doctor.dnp = form.dnp.data
        doctor.dnp_grad_month_year = form.dnp_grad_month_year.data
        doctor.additional_training = form.additional_training.data
        doctor.sponsorship_needed = bool(form.sponsorship_needed.data)

        # Malpractice
        n_cases = int(form.num_malpractice_cases.data)
        cases = []
        for entry in form.malpractice_cases.entries[:n_cases]:
            cases.append({
                "incident_year": entry.form.incident_year.data,
                "outcome": entry.form.outcome.data,
                "payout_amount": entry.form.payout_amount.data or 0
            })
        doctor.malpractice_cases = json.dumps(cases)

        # Certification
        doctor.certification = form.certification.data
        doctor.certification_specialty_area = form.certification_specialty_area.data

        # Clinical
        doctor.clinically_active = form.clinically_active.data
        doctor.last_clinically_active = (
            form.last_clinically_active.data
            if form.clinically_active.data == "No"
            else None
        )

        # Extras
        doctor.emr = form.emr.data
        doctor.languages = form.languages.data
        doctor.states_licensed = ",".join(form.states_licensed.data)
        doctor.states_willing_to_work = ",".join(form.states_willing_to_work.data)
        doctor.salary_expectations = form.salary_expectations.data or 0

        db.session.commit()
        flash("Doctor updated!", "success")
        return redirect(url_for('doctor_profile', doctor_id=doctor.id))

    # Populate initial values (same as doctor_edit_profile)
    if request.method == 'GET':
        # Reuse identical population logic for consistency
        return doctor_edit_profile_populate(form, doctor)

    return render_template("edit_doctor.html", form=form, doctor=doctor, zip=zip)
# ============================================================
#                 DASHBOARD ROUTING LOGIC
# ============================================================

    """Redirect user to correct dashboard based on role."""
    if current_user.role == "admin":
        return redirect(url_for('admin_dashboard'))
    if current_user.role == "client":
        return redirect(url_for('client_dashboard'))
    if current_user.role == "doctor":
        return redirect(url_for('doctor_dashboard'))

    flash("Unknown role.", "danger")
    return redirect(url_for('logout'))


# ============================================================
#                     ADMIN DASHBOARD
# ============================================================

@app.route('/admin/dashboard')
@login_required
def admin_dashboard():
    if current_user.role != "admin":
        flash("Unauthorized", "danger")
        return redirect(url_for('dashboard'))

    jobs = Job.query.order_by(Job.id.desc()).all()
    doctors = Doctor.query.order_by(Doctor.id.desc()).all()

    return render_template("admin_dashboard.html", jobs=jobs, doctors=doctors)


# ============================================================
#                  CLIENT DASHBOARD + CALENDAR
# ============================================================

    if current_user.role != "client":
        flash("Unauthorized", "danger")
        return redirect(url_for('dashboard'))

    # Calls the client scheduled
    calls = ScheduledCall.query.filter_by(scheduled_by_id=current_user.id).all()
    reschedule_requests = ScheduledCall.query.filter_by(
        scheduled_by_id=current_user.id,
        reschedule_requested=True
    ).all()

    events = []
    for call in calls:
        if call.canceled:
            color, status = "#dc3545", "Canceled"
        elif call.reschedule_requested:
            color, status = "#17a2b8", "Reschedule Requested"
        elif call.invite_status == "Pending":
            color, status = "#ffc107", "Pending Invite"
        else:
            color, status = "#28a745", "Accepted"

        events.append({
            "id": call.id,
            "title": f"Call with Dr. {call.doctor.first_name} {call.doctor.last_name}",
            "start": call.datetime.isoformat(),
            "color": color,
            "status": status
        })

    return render_template(
        "client_dashboard.html",
        events=events,
        reschedule_requests=reschedule_requests
    )


# ============================================================
#                  DOCTOR DASHBOARD + CALENDAR
# ============================================================

    if current_user.role != "doctor":
        flash("Unauthorized", "danger")
        return redirect(url_for('dashboard'))

    doctor = current_user.doctor

    calls = ScheduledCall.query.filter_by(doctor_id=doctor.id).all()
    pending_invites = ScheduledCall.query.filter_by(
        doctor_id=doctor.id,
        invite_status="Pending"
    ).all()

    events = []
    for call in calls:

        if call.canceled:
            color, status = "#ff4d4d", "Canceled"
        elif call.reschedule_requested:
            color, status = "#3788d8", "Reschedule Requested"
        elif call.invite_status == "Pending":
            color, status = "#ffc107", "Pending Invite"
        elif call.invite_status == "Accepted":
            color, status = "#28a745", "Accepted"
        else:
            color, status = "#6c757d", "Scheduled"

        events.append({
            "id": call.id,
            "title": f"Call with {call.scheduled_by.username}",
            "start": call.datetime.isoformat(),
            "color": color,
            "status": status
        })

    return render_template(
        "doctor_dashboard.html",
        doctor=doctor,
        events=events,
        pending_invites=pending_invites
    )


# ============================================================
#                 SCHEDULE CALL — CLIENT/ADMIN
# ============================================================

@app.route('/schedule_call', methods=['GET', 'POST'])
@login_required
def schedule_call():
    if current_user.role not in ["client", "admin"]:
        flash("Unauthorized", "danger")
        return redirect(url_for('dashboard'))

    form = ScheduledCallForm()
    doctors = Doctor.query.order_by(Doctor.last_name.asc()).all()
    form.doctor_id.choices = [(d.id, f"{d.first_name} {d.last_name} | {d.email}") for d in doctors]

    if form.validate_on_submit():
        call = ScheduledCall(
            doctor_id=form.doctor_id.data,
            scheduled_by_id=current_user.id,
            job_id=None,
            datetime=form.datetime.data,
            reason=form.reason.data,
            invite_status="Pending"
        )

        db.session.add(call)
        db.session.commit()

        # Send notification to doctor
        send_system_message(
            sender_id=current_user.id,
            recipient_id=call.doctor.user_id,
            content=f"You have a call invite scheduled on {call.datetime.strftime('%Y-%m-%d %H:%M')}.",
            message_type="invite"
        )

        flash("Call scheduled!", "success")

        return redirect(url_for(
            'admin_dashboard' if current_user.role == "admin" else 'client_dashboard'
        ))

    return render_template("schedule_call.html", form=form)


# ============================================================
#                  DOCTOR INVITE RESPONSE
# ============================================================

@app.route('/doctor/handle_invite/<int:call_id>', methods=['POST'])
@login_required
def doctor_handle_invite(call_id):
    if current_user.role != "doctor":
        flash("Unauthorized", "danger")
        return redirect(url_for('dashboard'))

    call = ScheduledCall.query.get_or_404(call_id)
    action = request.form.get("action")

    if call.doctor_id != current_user.doctor.id:
        flash("Unauthorized", "danger")
        return redirect(url_for('doctor_dashboard'))

    if action == "accept":
        call.invite_status = "Accepted"
        db.session.commit()

        send_system_message(
            sender_id=current_user.id,
            recipient_id=call.scheduled_by_id,
            content=f"Dr. {current_user.doctor.first_name} {current_user.doctor.last_name} accepted your invitation.",
            message_type="invite_response"
        )

        flash("Invite accepted!", "success")

    elif action == "decline":
        call.invite_status = "Declined"
        db.session.commit()

        send_system_message(
            sender_id=current_user.id,
            recipient_id=call.scheduled_by_id,
            content=f"Dr. {current_user.doctor.first_name} {current_user.doctor.last_name} declined your invitation.",
            message_type="invite_response"
        )

        flash("Invite declined.", "warning")

    return redirect(url_for('doctor_dashboard'))


# ============================================================
#                      DOCTOR CALL DETAILS
# ============================================================

    if current_user.role != "doctor":
        flash("Unauthorized", "danger")
        return redirect(url_for('dashboard'))

    call = ScheduledCall.query.get_or_404(call_id)

    if call.doctor_id != current_user.doctor.id:
        flash("Unauthorized", "danger")
        return redirect(url_for('doctor_dashboard'))

    if request.method == "POST":
        action = request.form.get("action")

        # ------------- CANCEL -------------
        if action == "cancel":
            call.canceled = True
            db.session.commit()

            send_system_message(
                sender_id=current_user.id,
                recipient_id=call.scheduled_by_id,
                content="Doctor cancelled the scheduled call.",
                message_type="invite_response"
            )

            flash("Call canceled.", "success")
            return redirect(url_for('doctor_dashboard'))

        # ------------ RESCHEDULE ----------
        if action == "reschedule":
            new_dt = request.form.get("reschedule_datetime")
            note = request.form.get("reschedule_note")

            call.reschedule_requested = True
            call.reschedule_note = note
            call.reschedule_datetime = datetime.strptime(new_dt, "%Y-%m-%dT%H:%M")
            db.session.commit()

            send_system_message(
                sender_id=current_user.id,
                recipient_id=call.scheduled_by_id,
                content=(
                    f"Doctor requested a reschedule: "
                    f"{call.reschedule_datetime.strftime('%Y-%m-%d %H:%M')} — {note}"
                ),
                message_type="reschedule_request"
            )

            flash("Reschedule request sent.", "success")
            return redirect(url_for('doctor_dashboard'))

    return render_template("doctor_call_details.html", call=call)



# ============================================================
#                       ADMIN ANALYTICS
# ============================================================

@app.route('/admin/analytics')
@login_required
def admin_analytics():
    if current_user.role != "admin":
        flash("Unauthorized access!", "danger")
        return redirect(url_for('dashboard'))

    jobs = Job.query.order_by(Job.id.desc()).all()
    job_data = []

    for job in jobs:
        messages = Message.query.filter_by(job_id=job.id, message_type='interest').all()

        interest_by_day = defaultdict(int)
        interested_doctors = []

        for msg in messages:
            day = msg.timestamp.strftime("%Y-%m-%d")
            interest_by_day[day] += 1

            if msg.doctor:
                interested_doctors.append({
                    "id": msg.doctor.id,
                    "name": f"{msg.doctor.first_name} {msg.doctor.last_name}",
                    "email": msg.doctor.email
                })

        job_data.append({
            "id": job.id,
            "title": job.title,
            "location": job.location,
            "salary": job.salary,
            "description": job.description,
            "interest_count": len(messages),
            "interested_doctors": interested_doctors,
            "interest_by_day": interest_by_day,
            "client_name": job.poster.username if job.poster else "Unknown",
            "client_email": job.poster.email if job.poster else ""
        })

    return render_template("admin_analytics.html", job_data=job_data)


# ============================================================
#                     CLIENT ANALYTICS
# ============================================================

@app.route('/client/analytics')
@login_required
def client_analytics():
    if current_user.role != "client":
        flash("Unauthorized access!", "danger")
        return redirect(url_for('dashboard'))

    jobs = Job.query.filter_by(poster_id=current_user.id).order_by(Job.id.desc()).all()
    job_data = []

    for job in jobs:
        messages = Message.query.filter_by(job_id=job.id, message_type='interest').all()

        interest_by_day = defaultdict(int)
        interested_doctors = []

        for msg in messages:
            day = msg.timestamp.strftime("%Y-%m-%d")
            interest_by_day[day] += 1

            if msg.doctor:
                interested_doctors.append({
                    "id": msg.doctor.id,
                    "name": f"{msg.doctor.first_name} {msg.doctor.last_name}",
                    "email": msg.doctor.email
                })

        job_data.append({
            "id": job.id,
            "title": job.title,
            "location": job.location,
            "salary": job.salary,
            "description": job.description,
            "interest_count": len(messages),
            "interested_doctors": interested_doctors,
            "interest_by_day": interest_by_day,
        })

    return render_template("client_analytics.html", job_data=job_data)


# ============================================================
#            CLIENT DOWNLOAD EXCEL — INTERESTED DOCTORS
# ============================================================

@app.route('/client/analytics/download/<int:job_id>')
@login_required
def download_job_applicants(job_id):
    job = Job.query.get_or_404(job_id)

    # clients may only download their own job data
    if current_user.role == "client" and job.poster_id != current_user.id:
        flash("Not authorized for this job.", "danger")
        return redirect(url_for('client_analytics'))

    if current_user.role not in ["client", "admin"]:
        flash("Unauthorized access!", "danger")
        return redirect(url_for('dashboard'))

    messages = Message.query.filter_by(job_id=job.id, message_type='interest').all()
    doctor_ids = {msg.doctor_id for msg in messages if msg.doctor_id}

    doctors = Doctor.query.filter(Doctor.id.in_(doctor_ids)).all()

    rows = []
    for doc in doctors:
        rows.append({
            "First Name": doc.first_name,
            "Last Name": doc.last_name,
            "Email": doc.email,
            "Specialty": doc.specialty,
            "Subspecialty": doc.subspecialty,
            "City": doc.city_of_residence,
            "Phone": doc.phone,
            "Alt Phone": doc.alt_phone,
            "Salary Expectations": doc.salary_expectations,
            "States Licensed": doc.states_licensed,
            "States Willing": doc.states_willing_to_work,
            "Clinically Active": doc.clinically_active,
            "Joined": doc.joined.strftime("%Y-%m-%d") if doc.joined else "",
            "Malpractice Cases": doc.malpractice_cases
        })

    df = pd.DataFrame(rows)
    output = BytesIO()
    df.to_excel(output, index=False)
    output.seek(0)

    sanitized_title = job.title.replace(" ", "_")
    return send_file(
        output,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        download_name=f"Applicants_{sanitized_title}.xlsx",
        as_attachment=True
    )


# ============================================================
#                   MY JOBS — CLIENT VIEW
# ============================================================

    if current_user.role != "client":
        flash("Unauthorized access!", "danger")
        return redirect(url_for('dashboard'))

    keyword = request.args.get("keyword", "").lower()
    location = request.args.get("location", "").lower()

    query = Job.query.filter_by(poster_id=current_user.id)

    if keyword:
        query = query.filter(
            Job.title.ilike(f"%{keyword}%") |
            Job.description.ilike(f"%{keyword}%")
        )

    if location:
        query = query.filter(Job.location.ilike(f"%{location}%"))

    jobs = query.order_by(Job.id.desc()).all()

    return render_template("client_my_jobs.html", jobs=jobs, keyword=keyword, location=location)


# ============================================================
#             DOCTOR PROFILE PAGE (PUBLIC TO CLIENTS)
# ============================================================

    doctor = Doctor.query.get_or_404(doctor_id)
    malpractice_cases = json.loads(doctor.malpractice_cases or "[]")

    return render_template(
        "doctor_profile.html",
        doctor=doctor,
        malpractice_cases=malpractice_cases
    )
# ============================================================
#                      MESSAGING SYSTEM
# ============================================================

def send_system_message(sender_id, recipient_id, content, message_type="system", job_id=None, doctor_id=None):
    """Reusable helper for internal notifications."""
    m = Message(
        sender_id=sender_id,
        recipient_id=recipient_id,
        content=content,
        message_type=message_type,
        job_id=job_id,
        doctor_id=doctor_id
    )
    db.session.add(m)
    db.session.commit()
    return m


# ============================================================
#                          INBOX
# ============================================================

@app.route("/inbox")
@login_required
def inbox():
    """Unified inbox: doctors, clients, admins all use this."""
    messages = Message.query.filter_by(recipient_id=current_user.id)\
                            .order_by(Message.timestamp.desc()).all()

    # Mark unread as read
    unread = [m for m in messages if not m.read]
    for m in unread:
        m.read = True

    if unread:
        db.session.commit()

    return render_template("inbox.html", messages=messages)


# ============================================================
#                          SENT MESSAGES
# ============================================================

@app.route("/sent")
@login_required
def sent_messages():
    messages = Message.query.filter_by(sender_id=current_user.id)\
                            .order_by(Message.timestamp.desc()).all()
    return render_template("sent_messages.html", messages=messages)


# ============================================================
#               VIEW INDIVIDUAL MESSAGE
# ============================================================

@app.route("/message/<int:message_id>")
@login_required
def view_message(message_id):
    msg = Message.query.get_or_404(message_id)

    if msg.recipient_id != current_user.id and msg.sender_id != current_user.id:
        flash("Unauthorized.", "danger")
        return redirect(url_for("inbox"))

    if msg.recipient_id == current_user.id and not msg.read:
        msg.read = True
        db.session.commit()

    return render_template("view_message.html", message=msg)


# ============================================================
#               SEND A MANUAL MESSAGE (CLIENT/DOCTOR)
# ============================================================

    recipient = User.query.get_or_404(recipient_id)
    form = MessageForm()

    if form.validate_on_submit():
        content = form.content.data.strip()
        if not content:
            flash("Message cannot be empty.", "danger")
            return redirect(request.url)

        msg = Message(
            sender_id=current_user.id,
            recipient_id=recipient.id,
            content=content,
            message_type="manual"
        )

        db.session.add(msg)
        db.session.commit()

        flash("Message sent!", "success")
        return redirect(url_for("sent_messages"))

    return render_template("send_message.html", form=form, recipient=recipient)


# ============================================================
#                 MESSAGE THREAD BETWEEN TWO USERS
# ============================================================

@app.route("/messages/thread/<int:user_id>")
@login_required
def message_thread(user_id):
    """View a 1-on-1 conversation thread."""
    other = User.query.get_or_404(user_id)

    thread = Message.query.filter(
        ((Message.sender_id == current_user.id) & (Message.recipient_id == other.id)) |
        ((Message.sender_id == other.id) & (Message.recipient_id == current_user.id))
    ).order_by(Message.timestamp.asc()).all()

    # Mark unread as read
    unread = [m for m in thread if m.recipient_id == current_user.id and not m.read]
    for m in unread:
        m.read = True

    if unread:
        db.session.commit()

    return render_template("message_thread.html", thread=thread, other=other)


# ============================================================
#        EXTERNAL CALLS TO SEND INTEREST MESSAGES
# ============================================================

    """Doctor expresses interest in a job (client gets notified)."""
    job = Job.query.get_or_404(job_id)

    if current_user.role != "doctor":
        flash("Unauthorized", "danger")
        return redirect(url_for("doctor_jobs"))

    send_system_message(
        sender_id=current_user.id,
        recipient_id=job.poster_id,
        content=f"Dr. {current_user.doctor.first_name} {current_user.doctor.last_name} expressed interest in your job: '{job.title}'.",
        message_type="interest",
        job_id=job.id,
        doctor_id=current_user.doctor.id
    )

    flash("Interest sent!", "success")
    return redirect(url_for("doctor_jobs"))
# ============================================================
#                     LANDING + AUTH PAGES
# ============================================================

    """Public landing page."""
    return render_template("landing_page.html", current_year=datetime.now().year)


    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    form = LoginForm()

    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data.strip()).first()

        if user and user.check_password(form.password.data):
            login_user(user)
            flash("Logged in successfully!", "success")
            return redirect(url_for("dashboard"))

        flash("Invalid username or password.", "danger")

    return render_template("login.html", form=form)


    logout_user()
    flash("Logged out successfully.", "success")
    return redirect(url_for("login"))


@app.route("/create_account")
def create_account():
    """Public page where users choose doctor or client signup."""
    return render_template("create_account.html")
# ============================================================
#               PUBLIC CLIENT REGISTRATION
# ============================================================

    data = request.form

    username = data.get("username", "").strip()
    email = data.get("email", "").strip()
    password = data.get("password", "")
    confirm = data.get("confirm_password", "")

    if password != confirm:
        flash("Passwords do not match.", "danger")
        return redirect(url_for("create_account"))

    # Check duplicates
    if User.query.filter((User.username == username) | (User.email == email)).first():
        flash("Username or email already exists.", "danger")
        return redirect(url_for("create_account"))

    # Create user
    user = User(username=username, email=email, role="client")
    user.set_password(password)

    db.session.add(user)
    db.session.commit()

    login_user(user)
    flash("Client account created!", "success")
    return redirect(url_for("client_dashboard"))
# ============================================================
#               PUBLIC DOCTOR REGISTRATION
# ============================================================

    data = request.form

    username = data.get("username", "").strip()
    email = data.get("email", "").strip()
    password = data.get("password", "")
    confirm = data.get("confirm_password", "")

    first_name = data.get("first_name", "").strip()
    last_name = data.get("last_name", "").strip()
    specialty = data.get("specialty", "").strip()

    if password != confirm:
        flash("Passwords do not match.", "danger")
        return redirect(url_for("create_account"))

    # Check duplicates
    if User.query.filter((User.username == username) | (User.email == email)).first():
        flash("Username or email already exists.", "danger")
        return redirect(url_for("create_account"))

    # Create user
    user = User(username=username, email=email, role="doctor")
    user.set_password(password)
    db.session.add(user)
    db.session.commit()

    # Create doctor profile linked to user
    doc = Doctor(
        first_name=first_name,
        last_name=last_name,
        email=email,
        specialty=specialty,
        position="MD",
        user_id=user.id,
        joined=datetime.utcnow()
    )

    db.session.add(doc)
    db.session.commit()

    login_user(user)
    flash("Doctor account created!", "success")
    return redirect(url_for("doctor_dashboard"))
# ============================================================
#           ADMIN-ONLY — CREATE INTERNAL USERS
# ============================================================

    if current_user.role != "admin":
        flash("Only admins may create internal users.", "danger")
        return redirect(url_for("dashboard"))

    form = RegisterForm()

    if form.validate_on_submit():
        username = form.username.data.strip()

        if User.query.filter_by(username=username).first():
            flash("Username already exists.", "danger")
            return redirect(url_for("register"))

        new_user = User(username=username)
        new_user.set_password(form.password.data)

        db.session.add(new_user)
        db.session.commit()

        flash("New internal user created!", "success")
        return redirect(url_for("admin_dashboard"))

    return render_template("register.html", form=form)
# ============================================================
#            FINAL APP INITIALIZATION & AUTO-SETUP
# ============================================================

with app.app_context():
    # Create tables if not exist
    db.create_all()

    # Ensure default admin exists
    admin_user = User.query.filter_by(username="adminchan").first()
    if not admin_user:
        admin_user = User(
            username="adminchan",
            email="admin@example.com",
            role="admin",
            password_hash=generate_password_hash("icecream2")
        )
        db.session.add(admin_user)
        db.session.commit()
        print("Admin account created: adminchan / icecream2")
    else:
        print("Admin account already exists.")


# ============================================================
#                  AUTO-OPEN BROWSER ON LAUNCH
# ============================================================

def open_browser():
    time.sleep(1)
    webbrowser.open("http://localhost:5000")


threading.Thread(target=open_browser).start()


# ============================================================
#                          RUN APP
# ============================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)












