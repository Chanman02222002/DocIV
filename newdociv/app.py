from flask import Flask, render_template, redirect, url_for, request, flash
import argparse
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, SelectMultipleField, FloatField
from wtforms.validators import DataRequired, Email
from datetime import datetime
import threading, webbrowser, time
from jinja2 import DictLoader
from wtforms.widgets import ListWidget, CheckboxInput
from flask import send_from_directory
from wtforms import SelectField
from wtforms import StringField, SubmitField, SelectMultipleField, FloatField, SelectField, BooleanField, FieldList, FormField, HiddenField
from wtforms.validators import DataRequired, Email, Optional
import json
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, login_user, logout_user, login_required, UserMixin, current_user
from flask import jsonify
from wtforms import SelectField, DateTimeLocalField, TextAreaField, SubmitField
import os
from wtforms.validators import DataRequired
from pathlib import Path
from werkzeug.serving import run_simple
from collections import defaultdict
from flask_login import current_user
from werkzeug.utils import secure_filename
from flask_wtf.file import FileField, FileAllowed
import openai
import re
from flask import jsonify, request
import html
from flask_login import login_required, current_user
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
from collections import defaultdict
from flask import send_file
import pandas as pd
from io import BytesIO
from sqlalchemy import inspect, text



print("Current working directory:", os.getcwd())
print("Database path:", os.path.abspath('crm.db'))
app = Flask(__name__, static_folder='static')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///crm.db'

from dotenv import load_dotenv
load_dotenv()
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'fallback-secret')

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.Text, nullable=False)
    role = db.Column(db.String(20), default='client')  # default as client

    doctor = db.relationship('Doctor', back_populates='user', uselist=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Job(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100))
    location = db.Column(db.String(100))
    salary = db.Column(db.String(50))
    description = db.Column(db.Text)
    job_url = db.Column(db.Text, nullable=True)
    date_posted = db.Column(db.String(50), nullable=True)
    poster_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    latitude = db.Column(db.Float, nullable=True)
    longitude = db.Column(db.Float, nullable=True)

    poster = db.relationship('User', backref='jobs')
    requirements = db.relationship('JobRequirement', back_populates='job', uselist=False, cascade="all, delete-orphan")


class JobRequirement(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    job_id = db.Column(db.Integer, db.ForeignKey('job.id'), unique=True)
    position = db.Column(db.String(10))
    specialty = db.Column(db.String(100))
    subspecialty = db.Column(db.String(100))
    certification = db.Column(db.String(50))
    certification_specialty_area = db.Column(db.String(100))
    clinically_active = db.Column(db.String(50))
    emr = db.Column(db.Text)
    emr_other = db.Column(db.String(255))
    languages = db.Column(db.Text)
    language_other = db.Column(db.String(255))
    states_required = db.Column(db.Text)
    states_preferred = db.Column(db.Text)
    sponsorship_supported = db.Column(db.Boolean, default=False)
    salary_range = db.Column(db.String(100))
    notes = db.Column(db.Text)

    job = db.relationship('Job', back_populates='requirements')



class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    sender_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    recipient_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    job_id = db.Column(db.Integer, db.ForeignKey('job.id'), nullable=True)
    doctor_id = db.Column(db.Integer, db.ForeignKey('doctor.id'), nullable=True)
    content = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    read = db.Column(db.Boolean, default=False)
    message_type = db.Column(db.String(50), default='general')  # <-- NEW FIELD

    sender = db.relationship('User', foreign_keys=[sender_id], backref='sent_messages')
    recipient = db.relationship('User', foreign_keys=[recipient_id], backref='received_messages')
    job = db.relationship('Job', backref='messages')
    doctor = db.relationship('Doctor', backref='messages')


def ensure_job_columns():
    """Add new job columns to existing databases without migrations."""
    with app.app_context():
        inspector = inspect(db.engine)
        if not inspector.has_table('job'):
            return
        existing = {col['name'] for col in inspector.get_columns('job')}
        statements = []

        if 'job_url' not in existing:
            statements.append("ALTER TABLE job ADD COLUMN job_url TEXT")
        if 'date_posted' not in existing:
            statements.append("ALTER TABLE job ADD COLUMN date_posted VARCHAR(50)")
        if 'latitude' not in existing:
            statements.append("ALTER TABLE job ADD COLUMN latitude FLOAT")
        if 'longitude' not in existing:
            statements.append("ALTER TABLE job ADD COLUMN longitude FLOAT")

        if statements:
            with db.engine.begin() as conn:
                for stmt in statements:
                    conn.execute(text(stmt))


ensure_job_columns()


# Doctor Registration Form (For Admin)
class DoctorRegistrationForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = StringField('Password', validators=[DataRequired()])
    submit = SubmitField('Create Doctor')

# Job Posting Form (For User)
class JobForm(FlaskForm):
    title = StringField('Title', validators=[DataRequired()])
    location = StringField('Location', validators=[DataRequired()])
    salary = StringField('Salary', validators=[DataRequired()])
    description = TextAreaField('Description', validators=[DataRequired()])
    submit = SubmitField('Post Job')

# Database Models Updates

class ClientRegistrationForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = StringField('Password', validators=[DataRequired()])
    submit = SubmitField('Create Client')

geolocator = Nominatim(user_agent="dociv_geocoder")

# Limit how many direct-job rows we geocode per sync to keep doctor_jobs responsive.
DIRECT_JOBS_GEOCODE_LIMIT = 3



def safe_geocode(location_str):
    """Return a geopy location object or None if lookup fails."""
    try:
        return geolocator.geocode(location_str, timeout=3)
    except (GeocoderTimedOut, GeocoderUnavailable, Exception):
        return None


def geocode_location(location_str):
    if not location_str:
        return None, None

    loc = safe_geocode(location_str)
    if loc:
        return loc.latitude, loc.longitude
    return None, None


def geocode_missing_jobs():
    """Backfill latitude/longitude for jobs that are missing coordinates."""
    with app.app_context():
        jobs = Job.query.all()
        updated = 0
        for job in jobs:
            if job.latitude is not None and job.longitude is not None:
                continue
            if not job.location:
                print(f"Skipping job {job.id}: no location provided")
                continue

            loc = safe_geocode(job.location)
            if loc:
                job.latitude = loc.latitude
                job.longitude = loc.longitude
                updated += 1
                print(f"Saved geocode for job {job.id}: {loc.latitude}, {loc.longitude}")
            else:
                print(f"Could not geocode job {job.id} ({job.location})")

        if updated:
            db.session.commit()
            print(f"Committed geocodes for {updated} job(s)")
        else:
            print("No jobs required geocoding")


def filter_jobs_by_specialty(jobs, specialty, subspecialty=None):
    """Filter jobs to those that mention the doctor's specialty or subspecialty."""
    if not specialty and not subspecialty:
        return jobs

    specialty_terms = [term.lower() for term in [specialty, subspecialty] if term]

    def matches(job):
        text = f"{job.title or ''} {job.description or ''}".lower()
        return any(term in text for term in specialty_terms)

    filtered = [job for job in jobs if matches(job)]
    return filtered or jobs


def parse_salary_value(salary_str):
    """Best-effort numeric extraction for salary sorting."""
    if not salary_str:
        return 0
    numbers = re.findall(r"[\d,]+", salary_str)
    if not numbers:
        return 0
    try:
        return float(numbers[0].replace(',', ''))
    except ValueError:
        return 0


def build_fallback_suggestions(jobs_payload, doctor_profile=None):
    """Return deterministic suggestions when AI is unavailable."""
    doctor_city = doctor_profile.get("home_base") if doctor_profile else ""
    licensed_states = set((doctor_profile or {}).get("licensed_states") or [])
    preferred_states = set((doctor_profile or {}).get("preferred_states") or [])

    def state_priority(job):
        job_state = extract_state_abbr(job.get("location"))
        if job_state in licensed_states:
            return 2
        if job_state in preferred_states:
            return 1
        return 0

    fallback_sorted = sorted(
        jobs_payload,
        key=lambda j: (
            state_priority(j),
            parse_salary_value(j.get("salary")),
        ),
        reverse=True,
    )

    suggestions = []
    for job in fallback_sorted:
        rationale_bits = ["Matches your specialty details"]
        job_state = extract_state_abbr(job.get("location"))
        if job_state in licensed_states:
            rationale_bits.append(f"licensed for {job_state}")
        elif job_state in preferred_states:
            rationale_bits.append(f"matches your interest in {job_state}")
        if doctor_city and job.get("location"):
            rationale_bits.append(f"location compared to {doctor_city}")
        if job.get("salary"):
            rationale_bits.append("competitive compensation noted")

        suggestions.append(
            {
                "id": job["id"],
                "title": job["title"],
                "location": job.get("location"),
                "salary": job.get("salary"),
                "rationale": ", ".join(rationale_bits) + ".",
                "score": max(50, 70 - len(suggestions)) + state_priority(job) * 10,
            }
        )

    return suggestions[:10]


def get_doctor_jobs_payload(doctor):
    """Return a list of job dictionaries scoped to a doctor's specialty and state priorities."""
    scoped_jobs = get_doctor_jobs(doctor)

    return [
        {
            "id": job.id,
            "title": job.title,
            "location": job.location,
            "salary": job.salary,
            "description": job.description,
        }
        for job in scoped_jobs
    ]


def get_doctor_jobs(doctor):
    """Return scoped Job objects filtered by specialty and state preferences."""
    raw_jobs = Job.query.order_by(Job.id.desc()).all()
    scoped_jobs = filter_jobs_by_specialty(raw_jobs, doctor.specialty, doctor.subspecialty)

    licensed_states = set(normalize_state_values((doctor.states_licensed or "").split(',')))
    preferred_states = set(normalize_state_values((doctor.states_willing_to_work or "").split(',')))

    if licensed_states or preferred_states:
        matching_state_jobs = []
        fallback_jobs = []

        for job in scoped_jobs:
            job_state = extract_state_abbr(job.location)
            if job_state in licensed_states or job_state in preferred_states:
                matching_state_jobs.append(job)
            else:
                fallback_jobs.append(job)

        if matching_state_jobs:
            scoped_jobs = matching_state_jobs
        else:
            scoped_jobs = fallback_jobs or scoped_jobs

    return scoped_jobs
def format_city_state(city, state):
    parts = []
    if city:
        parts.append(city.strip())
    if state:
        parts.append(state.strip())
    return ", ".join(parts)


def split_city_state(value):
    city, state = "", ""
    if value:
        parts = [p.strip() for p in value.split(',')]
        if parts:
            city = parts[0]
        if len(parts) > 1:
            state = parts[1]
    return city, state


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))
states = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]

STATE_NAME_TO_ABBR = {
    "ALABAMA": "AL",
    "ALASKA": "AK",
    "ARIZONA": "AZ",
    "ARKANSAS": "AR",
    "CALIFORNIA": "CA",
    "COLORADO": "CO",
    "CONNECTICUT": "CT",
    "DELAWARE": "DE",
    "FLORIDA": "FL",
    "GEORGIA": "GA",
    "HAWAII": "HI",
    "IDAHO": "ID",
    "ILLINOIS": "IL",
    "INDIANA": "IN",
    "IOWA": "IA",
    "KANSAS": "KS",
    "KENTUCKY": "KY",
    "LOUISIANA": "LA",
    "MAINE": "ME",
    "MARYLAND": "MD",
    "MASSACHUSETTS": "MA",
    "MICHIGAN": "MI",
    "MINNESOTA": "MN",
    "MISSISSIPPI": "MS",
    "MISSOURI": "MO",
    "MONTANA": "MT",
    "NEBRASKA": "NE",
    "NEVADA": "NV",
    "NEW HAMPSHIRE": "NH",
    "NEW JERSEY": "NJ",
    "NEW MEXICO": "NM",
    "NEW YORK": "NY",
    "NORTH CAROLINA": "NC",
    "NORTH DAKOTA": "ND",
    "OHIO": "OH",
    "OKLAHOMA": "OK",
    "OREGON": "OR",
    "PENNSYLVANIA": "PA",
    "RHODE ISLAND": "RI",
    "SOUTH CAROLINA": "SC",
    "SOUTH DAKOTA": "SD",
    "TENNESSEE": "TN",
    "TEXAS": "TX",
    "UTAH": "UT",
    "VERMONT": "VT",
    "VIRGINIA": "VA",
    "WASHINGTON": "WA",
    "WEST VIRGINIA": "WV",
    "WISCONSIN": "WI",
    "WYOMING": "WY",
}


def normalize_state_values(raw_states):
    """Return a de-duplicated list of state abbreviations from mixed inputs."""
    normalized = []
    for value in raw_states:
        if not value:
            continue
        token = re.sub(r"\s+", " ", str(value)).replace(".", "").strip().upper()
        if token in states:
            normalized.append(token)
        elif token in STATE_NAME_TO_ABBR:
            normalized.append(STATE_NAME_TO_ABBR[token])
        elif len(token) == 2 and token.isalpha():
            normalized.append(token)

    # Preserve order while removing duplicates
    seen = set()
    deduped = []
    for token in normalized:
        if token not in seen:
            seen.add(token)
            deduped.append(token)
    return deduped



# Models
def extract_state_abbr(location):
    """More accurate state extraction from job location strings."""
    if not location:
        return ""

    loc = location.strip().upper()

    # 1. Try exact ", XX" match
    match = re.search(r",\s*([A-Z]{2})\b", loc)
    if match:
        abbr = match.group(1)
        if abbr in states:
            return abbr

    # 2. Try last token as 2-letter state
    parts = loc.split()
    if parts:
        last = parts[-1]
        if last in states:
            return last

    # 3. Try full state name → convert to abbreviation
    for name, abbr in STATE_NAME_TO_ABBR.items():
        if name in loc:
            return abbr

    return ""


@app.context_processor
def inject_user():
    return dict(current_user=current_user)

    
class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = StringField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')

class RegisterForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = StringField('Password', validators=[DataRequired()])
    submit = SubmitField('Register')

    
class Doctor(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), unique=True, nullable=False)
    user = db.relationship('User', back_populates='doctor')

    position = db.Column(db.String(10), nullable=False)
    specialty = db.Column(db.String(100))
    subspecialty = db.Column(db.String(100))
    first_name = db.Column(db.String(50))
    last_name = db.Column(db.String(50))
    email = db.Column(db.String(100), unique=True)
    phone = db.Column(db.String(20))
    alt_phone = db.Column(db.String(20))
    address = db.Column(db.String(255))
    city_of_residence = db.Column(db.String(100))
    medical_school = db.Column(db.String(100))
    med_grad_month_year = db.Column(db.String(20))
    residency = db.Column(db.String(100))
    residency_grad_month_year = db.Column(db.String(20))
    fellowship = db.Column(db.Text)
    fellowship_grad_month_year = db.Column(db.Text)
    bachelors = db.Column(db.String(100))
    bachelors_grad_month_year = db.Column(db.String(20))
    msn = db.Column(db.String(100))
    msn_grad_month_year = db.Column(db.String(20))
    dnp = db.Column(db.String(100))
    dnp_grad_month_year = db.Column(db.String(20))
    additional_training = db.Column(db.Text)
    sponsorship_needed = db.Column(db.Boolean)
    malpractice_cases = db.Column(db.Text)
    certification = db.Column(db.String(30))
    certification_specialty_area = db.Column(db.String(100)) 
    clinically_active = db.Column(db.String(30))  # Yes, No, Never clinically active
    last_clinically_active = db.Column(db.String(20), nullable=True)
    emr = db.Column(db.String(100))
    languages = db.Column(db.String(200))
    states_licensed = db.Column(db.Text)
    states_willing_to_work = db.Column(db.Text)
    salary_expectations = db.Column(db.Float)
    joined = db.Column(db.DateTime, default=datetime.utcnow)
    profile_picture = db.Column(db.String(255), nullable=True)








class ScheduledCall(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    doctor_id = db.Column(db.Integer, db.ForeignKey('doctor.id'))
    scheduled_by_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    job_id = db.Column(db.Integer, db.ForeignKey('job.id'), nullable=True)
    datetime = db.Column(db.DateTime, nullable=False)
    reason = db.Column(db.String(255), nullable=True)
    canceled = db.Column(db.Boolean, default=False)
    reschedule_requested = db.Column(db.Boolean, default=False)
    reschedule_note = db.Column(db.String(255), nullable=True)
    reschedule_datetime = db.Column(db.DateTime, nullable=True)
    invite_status = db.Column(db.String(20), default="Pending") # New: Pending/Accepted/Declined

    # CRUCIAL FIX (MUST MATCH EXACTLY)
    doctor = db.relationship('Doctor', backref=db.backref('scheduled_calls', lazy=True))
    
    scheduled_by = db.relationship('User', backref=db.backref('scheduled_calls_scheduled', lazy=True))
    job = db.relationship('Job', backref='scheduled_calls') 

    def __repr__(self):
        return f'<ScheduledCall {self.id} with Doctor {self.doctor_id}>'






class MalpracticeCaseForm(FlaskForm):
    incident_year = StringField('Incident Year')
    outcome = SelectField(
        'Outcome',
        choices=[('Dropped', 'Dropped'), ('Won', 'Won'), ('Settled/Lost', 'Settled/Lost')],
        validators=[Optional()]
    )
    payout_amount = FloatField('Payout Amount', validators=[Optional()])
    case_explanation = TextAreaField('Case Explanation', validators=[Optional()])

    class Meta:
        csrf = False
# Forms
class DoctorForm(FlaskForm):
    position = SelectField('Healthcare Provider Type', choices=[('MD','MD'),('DO','DO'),('NP','NP'),('PA','PA')], validators=[DataRequired()])
    specialty_choices = [
        "Internal Medicine", "Family Medicine", "Pediatrics", "Emergency Medicine", "General Surgery",
        "Psychiatry", "Obstetrics and Gynecology", "Anesthesiology", "Diagnostic Radiology", "Pathology",
        "Orthopedic Surgery", "Cardiology", "Gastroenterology", "Dermatology", "Neurology", "Urology",
        "Otolaryngology (ENT)", "Ophthalmology", "Hematology/Oncology", "Physical Medicine and Rehabilitation (PM&R)",
        "Endocrinology", "Rheumatology", "Nephrology", "Pulmonology", "Infectious Disease", "Geriatrics",
        "Allergy and Immunology", "Plastic Surgery", "Cardiothoracic Surgery", "Vascular Surgery", "Neurosurgery",
        "Colorectal Surgery", "Bariatric Surgery", "Trauma Surgery", "Interventional Radiology",
        "Interventional Cardiology", "Pain Medicine", "Critical Care Medicine", "Sports Medicine", "Hospital Medicine",
        "Palliative Care", "Sleep Medicine", "Medical Genetics", "Occupational Medicine", "Preventive Medicine",
        "Nuclear Medicine", "Radiation Oncology", "Reproductive Endocrinology and Infertility (REI)",
        "Maternal-Fetal Medicine", "Adolescent Medicine", "Other"
    ]
    specialty = SelectField(
        'Specialty',
        choices=[(spec, spec) for spec in specialty_choices],
        validators=[DataRequired()],
        validate_choice=False
    )
    subspecialty = StringField('Subspecialty', validators=[Optional()])
    first_name = StringField('First Name', validators=[DataRequired()])
    last_name = StringField('Last Name', validators=[DataRequired()])
    email = StringField('Email', validators=[DataRequired(), Email()])
    phone = StringField('Phone Number', validators=[Optional()])
    alt_phone = StringField('Alternative Phone Number', validators=[Optional()])
    address = StringField('Address Line 1', validators=[Optional()])
    city = StringField('City', validators=[Optional()])
    state = SelectField('State', choices=[('', 'Select State')] + [(abbr, abbr) for abbr in states], validators=[Optional()])
    city_of_residence = HiddenField()
    # MD/DO fields
    medical_school = StringField('Medical School', validators=[Optional()])
    med_grad_month_year = StringField('Medical School Graduation (Month/Year)', validators=[Optional()])
    residency = StringField('Residency', validators=[Optional()])
    residency_grad_month_year = StringField('Residency Graduation (Month/Year)', validators=[Optional()])

    num_fellowships = SelectField(
        'Number of Fellowships',
        choices=[(str(i), str(i)) for i in range(0, 11)],
        validators=[DataRequired()]
    )
    fellowship = FieldList(
        StringField('Fellowship'),
        min_entries=1,
        max_entries=10
    )
    fellowship_grad_month_year = FieldList(
        StringField('Fellowship Graduation (Month/Year)'),
        min_entries=1,
        max_entries=10
    )

    # NP/PA fields
    bachelors = StringField('Bachelors Degree', validators=[Optional()])
    bachelors_grad_month_year = StringField('Bachelors Graduation (Month/Year)', validators=[Optional()])
    msn = StringField('Masters of Science in Nursing', validators=[Optional()])
    msn_grad_month_year = StringField('MSN Graduation (Month/Year)', validators=[Optional()])
    dnp = StringField('Doctor of Nursing', validators=[Optional()])
    dnp_grad_month_year = StringField('DNP Graduation (Month/Year)', validators=[Optional()])
    additional_training = StringField('Additional Training', validators=[Optional()])
    sponsorship_needed = BooleanField('Sponsorship Needed?', validators=[Optional()])

    num_malpractice_cases = SelectField(
        'Number of Malpractice Cases',
        choices=[(str(i), str(i)) for i in range(0, 21)],
        validators=[DataRequired()]
    )

    malpractice_cases = FieldList(
        FormField(MalpracticeCaseForm),
        min_entries=1,
        max_entries=15
    )

    certification = SelectField(
        'Certification',
        choices=[
            ('Board Certified', 'Board Certified'),
            ('Board Eligible', 'Board Eligible'),
            ('Not Boarded', 'Not Boarded')
        ],
        validators=[Optional()]
    )
    certification_specialty_area = StringField('Certification Specialty Area', validators=[Optional()])

    clinically_active = SelectField(
        'Clinically Active?',
        choices=[
            ('Yes', 'Yes'),
            ('No', 'No'),
            ('Never clinically active', 'Never clinically active')
        ],
        validators=[DataRequired()]
    )

    last_clinically_active = StringField('Last Clinically Active (Month/Year)', validators=[Optional()])


    emr_choices = [
        "Epic", "Cerner", "Allscripts", "Meditech", "NextGen", "athenahealth", "eClinicalWorks", "Practice Fusion",
        "GE Centricity", "Greenway Health", "McKesson", "Kareo", "Amazing Charts", "DrChrono", "AdvancedMD",
        "eMDs", "Aprima", "ChartLogic", "Other"
    ]
    emr = SelectMultipleField(
        'EMR Systems',
        choices=[(system, system) for system in emr_choices],
        validators=[Optional()],
        validate_choice=False,
        option_widget=CheckboxInput(),
        widget=ListWidget(prefix_label=False)
    )
    emr_other = StringField('Other EMR System', validators=[Optional()])
    language_choices = [
        "English", "Mandarin Chinese", "Hindi", "Spanish", "French", "Arabic", "Bengali", "Russian",
        "Portuguese", "Urdu", "Indonesian", "German", "Japanese", "Swahili", "Marathi", "Telugu",
        "Turkish", "Tamil", "Western Punjabi", "Korean", "Other"
    ]
    languages = SelectMultipleField(
        'Languages',
        choices=[(lang, lang) for lang in language_choices],
        validators=[Optional()],
        validate_choice=False,
        option_widget=CheckboxInput(),
        widget=ListWidget(prefix_label=False)
    )
    language_other = StringField('Other Language(s)', validators=[Optional()])

    states_licensed = SelectMultipleField(
        'States Licensed',
        choices=[(state, state) for state in states],
        option_widget=CheckboxInput(),
        widget=ListWidget(prefix_label=False)
    )
    states_willing_to_work = SelectMultipleField(
        'States Willing to Work',
        choices=[(state, state) for state in states],
        option_widget=CheckboxInput(),
        widget=ListWidget(prefix_label=False)
    )

    salary_expectations = FloatField('Salary Expectation (Total Compensation)', validators=[Optional()])
    profile_picture = FileField('Profile Picture', validators=[
        FileAllowed(['jpg', 'png', 'jpeg'], 'Images only!')
    ])

    submit = SubmitField('Submit')


class JobRequirementForm(FlaskForm):
    position = SelectField('Healthcare Provider Type', choices=[('MD','MD'),('DO','DO'),('NP','NP'),('PA','PA')], validators=[DataRequired()])
    specialty = SelectField(
        'Specialty',
        choices=[(spec, spec) for spec in DoctorForm.specialty_choices],
        validators=[DataRequired()],
        validate_choice=False
    )
    subspecialty = StringField('Subspecialty', validators=[Optional()])
    certification = SelectField(
        'Certification',
        choices=[
            ('Board Certified', 'Board Certified'),
            ('Board Eligible', 'Board Eligible'),
            ('Not Boarded', 'Not Boarded')
        ],
        validators=[Optional()]
    )
    certification_specialty_area = StringField('Certification Specialty Area', validators=[Optional()])
    clinically_active = SelectField(
        'Clinically Active?',
        choices=[
            ('Yes', 'Yes'),
            ('No', 'No'),
            ('Never clinically active', 'Never clinically active')
        ],
        validators=[Optional()]
    )
    emr = SelectMultipleField(
        'Preferred EMR Experience',
        choices=[(system, system) for system in DoctorForm.emr_choices],
        validators=[Optional()],
        validate_choice=False,
        option_widget=CheckboxInput(),
        widget=ListWidget(prefix_label=False)
    )
    emr_other = StringField('Other EMR System', validators=[Optional()])
    languages = SelectMultipleField(
        'Languages Needed',
        choices=[(lang, lang) for lang in DoctorForm.language_choices],
        validators=[Optional()],
        validate_choice=False,
        option_widget=CheckboxInput(),
        widget=ListWidget(prefix_label=False)
    )
    language_other = StringField('Other Language(s)', validators=[Optional()])
    states_required = SelectMultipleField(
        'States Required',
        choices=[(state, state) for state in states],
        option_widget=CheckboxInput(),
        widget=ListWidget(prefix_label=False)
    )
    states_preferred = SelectMultipleField(
        'Preferred States',
        choices=[(state, state) for state in states],
        option_widget=CheckboxInput(),
        widget=ListWidget(prefix_label=False)
    )
    sponsorship_supported = BooleanField('Sponsorship Available', validators=[Optional()])
    salary_range = StringField('Salary Range / Budget', validators=[Optional()])
    notes = TextAreaField('Additional Notes', validators=[Optional()])
    submit = SubmitField('Save Requirements')


class ScheduledCallForm(FlaskForm):
    doctor_id = SelectField('Doctor', validators=[DataRequired()])
    datetime = DateTimeLocalField('Call Date & Time', validators=[DataRequired()], format='%Y-%m-%dT%H:%M')
    reason = TextAreaField('Reason for Call', validators=[DataRequired()])
    submit = SubmitField('Schedule Call')
# Add this clearly above your route definitions:

app.jinja_loader = DictLoader({
    'base.html': '''
    <!doctype html>
    <html lang="en">

    <head>
        <title>Healthcare Systems</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/flatpickr/4.6.13/flatpickr.min.css">
        <link href='https://cdn.jsdelivr.net/npm/fullcalendar@6.1.11/index.global.min.css' rel='stylesheet' />
        <script src='https://cdn.jsdelivr.net/npm/fullcalendar@6.1.11/index.global.min.js'></script>
        <link href="https://cdnjs.cloudflare.com/ajax/libs/cropperjs/1.5.13/cropper.min.css" rel="stylesheet"/>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/cropperjs/1.5.13/cropper.min.js"></script>
        <style>
            :root {
                --brand-teal: #8ecad4;
                --brand-teal-dark: #5aa4b3;
                --brand-teal-soft: #e8f5f7;
            }

            body {
                background-color: #ffffff;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }

            .form-control,
            .form-select {
                border-radius: 10px;
                border: 1px solid #ced4da;
                padding: 12px;
                box-shadow: none;
                transition: border-color 0.3s ease;
            }

            .form-control:focus,
            .form-select:focus {
                border-color: var(--brand-teal-dark);
                box-shadow: 0 0 0 0.15rem rgba(90, 164, 179, 0.4);
            }
            label {
                font-weight: 600;
                color: #333;
            }

            .btn {
                border-radius: 50px;
                padding: 10px 20px;
            }

            .navbar {
                background-color: #121212 !important;
                box-shadow: none !important;
            }

            .navbar-brand img {
                height: 50px !important;
                width: auto;
                margin-right: 10px;
            }

            .navbar-brand {
                font-size: 1.4rem;
                font-weight: 600;
            }

            .navbar-nav .nav-link {
                font-size: 0.95rem;
                font-weight: 500;
                margin-left: 15px;
            }

            h1, h2, h3, h4 {
                font-weight: 700;
                color: #121212;
            }

            .footer {
                background-color: #f8f9fa;
                padding: 30px 0;
                margin-top: 50px;landing
            }

            .footer a {
                color: #333;
                text-decoration: none;
            }

            .footer a:hover {
                color: var(--brand-teal-dark);

            .alert {
                border-radius: 10px;
            }
            .leaflet-marker-icon-numbered {
                position: relative;
                width: 38px;
                height: 50px;
                background: transparent;
            }
            .leaflet-marker-icon-numbered .pin-img {
                width: 38px;
                height: 50px;
                display: block;
            }
            .leaflet-marker-icon-numbered .marker-badge {
                position: absolute;
                top: 6px;         /* Adjust down for vertical center of pin head */
                left: 5px;        /* Adjust for horizontal center of pin head */
                background: #fff;
                color: var(--brand-teal-dark);
                border: 2px solid var(--brand-teal-dark);
                border-radius: 50%;
                width: 20px;
                height: 20px;
                text-align: center;
                font-weight: bold;
                font-size: 1em;
                line-height: 16px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.1);
                z-index: 2;
                padding-top: 2px;
            }
            .leaflet-popup-content {
                padding: 0 !important;
                margin: 0 !important;
            }
            .custom-popup {
                max-width: 340px;
                min-width: 220px;
                border-radius: 14px;
                background: #fff;
                box-shadow: 0 8px 36px rgba(0,0,0,0.09), 0 1.5px 6px rgba(90, 164, 179, 0.16);
                padding: 0;
                overflow: hidden;
            }
            .custom-popup-header {
                background: var(--brand-teal-dark);
                color: #fff;
                font-weight: bold;
                padding: 14px 18px;
                font-size: 1.08em;
                border-bottom: 1.5px solid #eaf2fb;
            }
            
            .custom-job-list {
                max-height: 230px;    /* This enables scrolling for long lists */
                overflow-y: auto;
                padding: 10px 0 10px 0;
            }
            .custom-job {
                padding: 9px 18px 9px 18px;
                border-bottom: 1px solid #eaf2fb;
            }
            .custom-job:last-child {
                border-bottom: none;
            }
            .custom-job-title {
                font-weight: 600;
                color: var(--brand-teal-dark);
                font-size: 1em;
                margin-bottom: 0;
            }
            .custom-job-salary {
                color: #333;
                font-size: 0.97em;
            }
            .custom-view-job {
                color: var(--brand-teal-dark);
                font-weight: bold;
                text-decoration: underline;
                font-size: 0.97em;
                margin-top: 4px;
                display: inline-block;
            }
        </style>
    </head>

    <body>
        <nav class="navbar navbar-expand-lg navbar-dark">
            <div class="container">
                <a class="navbar-brand d-flex align-items-center" href="{{ url_for('landing_page') if request.endpoint == 'login' else url_for('dashboard') }}">
                    <img src="{{ url_for('static', filename='jobsdirectmedicalcutright.png') }}" alt="Logo">
                
                </a>
                <div class="navbar-nav ms-auto">
                    {% if current_user.is_authenticated %}

                        {% set unread_count = current_user.received_messages|selectattr('read', 'equalto', False)|list|length %}
                        {% if current_user.role == 'doctor' %}
                            <a class="nav-link text-white" href="{{ url_for('doctor_dashboard') }}">Dashboard</a>
                            <a class="nav-link text-white" href="{{ url_for('doctor_edit_profile') }}">Edit Profile</a>
                            <a class="nav-link text-white" href="{{ url_for('doctor_jobs') }}">Jobs</a>
                            <a class="nav-link text-white" href="{{ url_for('doctor_dashboard') }}#calendar-card">Calendar</a>
                            <a class="nav-link text-white position-relative" href="{{ url_for('doctor_inbox') }}">
                                Inbox
                                {% if unread_count > 0 %}
                                <span class="position-absolute top-0 start-100 translate-middle badge rounded-pill bg-danger">
                                    {{ unread_count }}
                                </span>
                                {% endif %}
                            </a>
                            <a class="nav-link text-white" href="{{ url_for('logout') }}">Logout</a>
                            {% if current_user.doctor.profile_picture %}
                                <a href="{{ url_for('doctor_edit_profile') }}">
                                    <img src="{{ url_for('static', filename=current_user.doctor.profile_picture) }}"
                                        alt="Profile"
                                        class="rounded-circle ms-3"
                                        style="height: 40px; width: 40px; object-fit: cover; border: 2px solid #fff;">
                                </a>
                                {% endif %}

                        
    
                            
                        {% elif current_user.role == 'admin' %}
                            <a class="nav-link text-white" href="{{ url_for('register_doctor') }}">Create Doctor Login</a>
                            <a class="nav-link text-white" href="{{ url_for('register_client') }}">Create Client Login</a>
                            <a class="nav-link text-white" href="{{ url_for('add_doctor') }}">Add Doctor</a>
                            <a class="nav-link text-white" href="{{ url_for('doctors') }}">View Doctors</a>
                            <a class="nav-link text-white" href="{{ url_for('post_job') }}">Post Job</a>
                            <a class="nav-link text-white" href="{{ url_for('admin_analytics') }}">Analytics</a>
                            <a class="nav-link text-white position-relative" href="{{ url_for('admin_inbox') }}">
                                Inbox
                                {% if unread_count > 0 %}
                                <span class="position-absolute top-0 start-100 translate-middle badge rounded-pill bg-danger">
                                    {{ unread_count }}
                                </span>
                                {% endif %}
                            </a>
                            <a class="nav-link text-white" href="{{ url_for('logout') }}">Logout</a>
                        {% elif current_user.role == 'client' %}
                            <a class="nav-link text-white" href="{{ url_for('post_job') }}">Post Job</a>
                            <a class="nav-link text-white" href="{{ url_for('client_my_jobs') }}">My Jobs</a>
                            <a class="nav-link text-white" href="{{ url_for('schedule_call') }}">Schedule Call</a>
                            <a class="nav-link text-white" href="{{ url_for('calls') }}">Scheduled Calls</a>
                            <a class="nav-link text-white" href="{{ url_for('doctors') }}">View Doctors</a>
                            <a class="nav-link text-white position-relative" href="{{ url_for('client_inbox') }}">
                                Inbox
                                {% if unread_count > 0 %}
                                <span class="position-absolute top-0 start-100 translate-middle badge rounded-pill bg-danger">
                                    {{ unread_count }}
                                </span>
                                {% endif %}
                            </a>
                            <a class="nav-link text-white" href="{{ url_for('logout') }}">Logout</a>
                        {% else %}
                            <a class="nav-link text-white" href="{{ url_for('logout') }}">Logout</a>
                        {% endif %}
                    {% else %}
                        <a class="nav-link text-white" href="{{ url_for('login') }}">Login</a>
                        <a class="nav-link text-white" href="{{ url_for('create_account') }}">Create Account</a>
                    {% endif %}
                </div>
            </div>
        </nav>

        <main class="container py-5">
            {% with messages = get_flashed_messages(with_categories=true) %}
            {% for category, msg in messages %}
            <div class="alert alert-{{ category }}">{{ msg }}</div>
            {% endfor %}
            {% endwith %}
            {% block content %}{% endblock %}
        </main>

        <footer class="footer text-center">
            <div class="container">
                <span>&copy; {{ current_year }} BECA Staffing Solutions. All rights reserved.</span>
            </div>
        </footer>

        <script src="https://cdn.jsdelivr.net/npm/flatpickr"></script>
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
        <script>
            flatpickr("#datetime", { enableTime: true, dateFormat: "Y-m-d\\TH:i" });
        </script>
    </body>

    </html>
    ''',



    'index.html': '''{% extends "base.html" %}{% block content %}
        <h1>Beca Staffing CRM Dashboard</h1>
        <a class="btn btn-success" href="/add_doctor">Add Doctor</a>
        <a class="btn btn-info" href="/schedule_call">Schedule Call with Doctor</a>
        <a class="btn btn-primary" href="/calls">View Scheduled Calls</a>
        <a class="btn btn-secondary" href="/doctors">View Doctors</a>
    {% endblock %}''',

    'login.html': '''{% extends "base.html" %}
        {% block content %}

        <style>
            :root {
                --brand-teal: #8ecad4;
                --brand-teal-dark: #5aa4b3;
                --brand-teal-soft: #e8f5f7;
            }

            body {
                background: linear-gradient(120deg, var(--brand-teal-soft) 0%, #f8f9fa 100%);
            }
            .login-container {
                max-width: 420px;
                margin: 70px auto 0 auto;
                background: #fff;
                border-radius: 20px;
                box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.18);
                padding: 38px 38px 30px 38px;
                display: flex;
                flex-direction: column;
                align-items: center;
                animation: fadeIn 1s;
            }
            .login-logo {
                width: 270px;
                height: 270px;
                object-fit: contain;
                margin-bottom: 2px;
            }
            .login-title {
                font-weight: 700;
                font-size: 2rem;
                margin-bottom: 12px;
                color: var(--brand-teal-dark);
            }
            .login-subtext {
                color: #555;
                margin-bottom: 24px;
                font-size: 1.08rem;
                text-align: center;
            }
            .form-control {
                border-radius: 12px;
                padding: 12px;
                font-size: 1.08em;
                margin-bottom: 18px;
                background: #f6f8fc;
                border: 1px solid #cfd8dc;
                transition: border-color 0.2s;
            }
            .form-control:focus {
                border-color: var(--brand-teal-dark);
                box-shadow: 0 0 0 0.09rem rgba(90, 164, 179, 0.35);
            }
            .btn-login {
                background: var(--brand-teal-dark);
                color: #fff;
                padding: 13px 0;
                width: 100%;
                border-radius: 50px;
                font-weight: 600;
                font-size: 1.18em;
                box-shadow: 0 2px 8px rgba(90, 164, 179, 0.25);
                margin-top: 10px;
                margin-bottom: 6px;
                transition: background 0.15s;
            }
            .btn-login:hover {
                background: #4c8f9f;
            }
            .login-links {
                display: flex;
                justify-content: space-between;
                width: 100%;
                margin-top: 12px;
            }
            .login-links a {
                font-size: 0.97em;
                color: var(--brand-teal-dark);
                text-decoration: none;
                transition: color 0.15s;
            }
            .login-links a:hover {
                color: #4c8f9f;
                text-decoration: underline;
            }
            @media (max-width: 540px) {
                .login-container { padding: 22px 8px 18px 8px; }
            }
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(30px);}
                to   { opacity: 1; transform: translateY(0);}
            }
        </style>

        <div class="login-container">
            <img src="{{ url_for('static', filename='jobsdirectmedicalcutright.png') }}" alt="Logo" class="login-logo">
            <div class="login-title">Welcome Back</div>
            <div class="login-subtext">Log in to your account to access your dashboard.</div>

            <form method="post">
                {{ form.hidden_tag() }}
                <div class="mb-3">
                    {{ form.username.label(class="form-label") }}
                    {{ form.username(class="form-control", placeholder="Username") }}
                </div>
                <div class="mb-3">
                    {{ form.password.label(class="form-label") }}
                    {{ form.password(class="form-control", placeholder="Password", type="password") }}
                </div>
                {{ form.submit(class="btn btn-login") }}
            </form>

            <div class="login-links mt-2">
                <a href="{{ url_for('create_account') }}">Create Account</a>
                <a href="#">Forgot Password?</a>
            </div>
        </div>

        {% endblock %}''',

    'client_dashboard.html': '''
        {% extends 'base.html' %}

        {% block content %}
        <style>
            .client-dashboard { color: #0f172a; }
            .client-dashboard .glass-card {
                background: linear-gradient(145deg, #f9fbff, #eef4ff);
                border: 1px solid #dbe7ff;
                box-shadow: 0 12px 28px rgba(17, 24, 39, 0.08);
                border-radius: 18px;
            }
            .client-dashboard .hero-card {
                background: radial-gradient(circle at 18% 15%, rgba(14, 165, 233, 0.1), transparent 35%),
                            radial-gradient(circle at 80% 0%, rgba(52, 211, 153, 0.18), transparent 40%),
                            linear-gradient(120deg, #f2f6ff, #e7f7ff);
                border: 1px solid #d7e7ff;
            }
            .client-dashboard .badge-soft-primary { background: #e0edff; color: #1d4ed8; }
            .client-dashboard .badge-soft-success { background: #e8f7ef; color: #15803d; }
            .client-dashboard .badge-soft-info { background: #e0f2fe; color: #0ea5e9; }
            .client-dashboard .nav-link { color: #1f2a44; border-radius: 999px; }
            .client-dashboard .nav-link.active, .client-dashboard .nav-link:hover { color: #0b3a82; background: #e0edff; }
            .stat-pill { border: 1px solid #dbe7ff; background: #fff; border-radius: 12px; padding: 10px 12px; }
            .activity-item { padding: 14px; border-radius: 12px; background: #fff; border: 1px solid #e5e7eb; }
            .activity-item + .activity-item { margin-top: 12px; }
            .reschedule-card form button { min-width: 96px; }
            .empty-state { color: #6b7280; }
        </style>

        <div class="client-dashboard">
            <div class="glass-card hero-card d-flex flex-column flex-lg-row justify-content-between align-items-start align-items-lg-center mb-4 p-4">
                <div>
                    <div class="text-uppercase small text-primary mb-2 fw-semibold">Client dashboard</div>
                    <h2 class="fw-bold mb-2">Welcome back, {{ current_user.username }}</h2>
                    <div class="d-flex flex-wrap gap-2">
                        <span class="badge rounded-pill badge-soft-primary fw-semibold">{{ total_jobs }} open roles</span>
                        <span class="badge rounded-pill badge-soft-success fw-semibold">{{ active_calls }} upcoming calls</span>
                        <span class="badge rounded-pill badge-soft-info fw-semibold">{{ total_interest }} interests logged</span>
                    </div>
                </div>
                <div class="d-flex flex-wrap gap-2 mt-3 mt-lg-0">
                    <a class="btn btn-primary" href="{{ url_for('post_job') }}">Post a job</a>
                    <a class="btn btn-outline-primary" href="{{ url_for('schedule_call') }}">Schedule call</a>
                    <a class="btn btn-outline-secondary" href="{{ url_for('client_analytics') }}">View analytics</a>
                </div>
            </div>

            <ul class="nav nav-pills mb-4 gap-2 dashboard-tabs">
                <li class="nav-item"><a class="nav-link active" href="#activity-section">Hiring activity</a></li>
                <li class="nav-item"><a class="nav-link" href="#calendar-card">Calendar</a></li>
                <li class="nav-item"><a class="nav-link" href="#inbox-section">Inbox</a></li>
            </ul>

            <div class="row g-4 align-items-stretch">
                <div class="col-lg-8" id="activity-section">
                    <div class="card glass-card h-100">
                        <div class="card-header d-flex justify-content-between align-items-center px-4 py-3">
                            <div>
                                <div class="text-uppercase small text-primary fw-semibold">Hiring activity</div>
                                <h5 class="mb-0">Recent interest on your roles</h5>
                            </div>
                            <a class="btn btn-sm btn-outline-primary" href="{{ url_for('client_my_jobs') }}">Manage roles</a>
                        </div>
                        <div class="card-body p-4">
                            <div class="d-flex flex-wrap gap-2 mb-3">
                                <div class="stat-pill"><div class="small text-muted">Roles posted</div><div class="fw-bold h5 mb-0">{{ total_jobs }}</div></div>
                                <div class="stat-pill"><div class="small text-muted">Total interest</div><div class="fw-bold h5 mb-0">{{ total_interest }}</div></div>
                                <div class="stat-pill"><div class="small text-muted">Upcoming calls</div><div class="fw-bold h5 mb-0">{{ active_calls }}</div></div>
                            </div>
                            {% if job_interest_summary %}
                                {% for job in job_interest_summary[:4] %}
                                    <div class="activity-item d-flex justify-content-between align-items-start gap-3">
                                        <div>
                                            <div class="fw-semibold">{{ job.title }}</div>
                                            <div class="text-muted small">{{ job.location }}</div>
                                        </div>
                                        <span class="badge bg-primary-subtle text-primary">{{ job.interest_count }} interested</span>
                                    </div>
                                {% endfor %}
                            {% else %}
                                <div class="empty-state">No roles posted yet. Post your first job to start receiving interest.</div>
                            {% endif %}
                        </div>
                        <div class="card-footer d-flex justify-content-between align-items-center px-4 py-3">
                            <div class="text-muted small">Track interests and schedule calls directly from your dashboard.</div>
                            <a class="btn btn-sm btn-outline-secondary" href="{{ url_for('doctors') }}">Browse doctors</a>
                        </div>
                    </div>
                </div>

                <div class="col-lg-4 d-flex flex-column gap-4">
                    <div class="card glass-card calendar-card" id="calendar-card">
                        <div class="card-header d-flex justify-content-between align-items-center px-4 py-3">
                            <div>
                                <div class="text-uppercase small text-primary fw-semibold">Calendar</div>
                                <h6 class="mb-0">Scheduled calls</h6>
                            </div>
                            <a class="btn btn-sm btn-outline-primary" href="{{ url_for('calls') }}">View all</a>
                        </div>
                        <div class="card-body p-3">
                            <div id="client-mini-calendar"></div>
                            <div class="mt-3">
                                {% if upcoming_calls %}
                                    {% for call in upcoming_calls[:3] %}
                                        <div class="d-flex justify-content-between align-items-start mb-2">
                                            <div>
                                                <div class="fw-semibold">Dr. {{ call.doctor.first_name }} {{ call.doctor.last_name }}</div>
                                                <div class="text-muted small">{{ call.datetime.strftime('%b %d, %Y %I:%M %p') }}</div>
                                            </div>
                                            <a class="btn btn-sm btn-outline-secondary" href="{{ url_for('edit_call', call_id=call.id) }}">Edit</a>
                                        </div>
                                    {% endfor %}
                                {% else %}
                                    <div class="empty-state">No upcoming calls scheduled.</div>
                                {% endif %}
                            </div>
                        </div>
                    </div>

                    <div class="card glass-card reschedule-card">
                        <div class="card-header px-4 py-3">
                            <div class="text-uppercase small text-primary fw-semibold">Reschedule requests</div>
                            <h6 class="mb-0">Pending decisions</h6>
                        </div>
                        <div class="card-body p-4">
                            {% if reschedule_requests %}
                                {% for request in reschedule_requests %}
                                    <div class="mb-4 pb-3 border-bottom">
                                        <div class="fw-semibold">Dr. {{ request.doctor.first_name }} {{ request.doctor.last_name }}</div>
                                        <div class="text-muted small">Original: {{ request.datetime.strftime('%b %d, %Y %I:%M %p') }}</div>
                                        <div class="text-muted small">Requested: {{ request.reschedule_datetime.strftime('%b %d, %Y %I:%M %p') }}</div>
                                        <div class="mt-2">{{ request.reschedule_note }}</div>
                                        <form action="{{ url_for('client_handle_reschedule', call_id=request.id) }}" method="post" class="mt-3 d-flex flex-wrap gap-2">
                                            <textarea name="client_note" class="form-control" placeholder="Optional note"></textarea>
                                            <div class="d-flex gap-2">
                                                <button type="submit" name="action" value="accept" class="btn btn-success">Accept</button>
                                                <button type="submit" name="action" value="decline" class="btn btn-outline-secondary">Decline</button>
                                            </div>
                                        </form>
                                    </div>
                                {% endfor %}
                            {% else %}
                                <div class="empty-state">No reschedule requests at this time.</div>
                            {% endif %}
                        </div>
                    </div>
                </div>
            </div>

            <div class="card glass-card mt-4" id="inbox-section">
                <div class="card-header d-flex justify-content-between align-items-center px-4 py-3">
                    <div>
                        <div class="text-uppercase small text-primary fw-semibold">Inbox</div>
                        <h5 class="mb-0">Latest conversations</h5>
                    </div>
                    <a class="btn btn-sm btn-outline-primary" href="{{ url_for('client_inbox') }}">Open inbox</a>
                </div>
                <div class="card-body p-4">
                    {% if message_preview %}
                        {% for message in message_preview %}
                            <div class="activity-item">
                                <div class="d-flex justify-content-between align-items-start gap-3">
                                    <div>
                                        <div class="fw-semibold">{{ message.sender.username if message.sender else 'System' }}</div>
                                        <div class="text-muted">{{ message.content[:180] }}{% if message.content|length > 180 %}...{% endif %}</div>
                                    </div>
                                    <small>{{ message.timestamp.strftime('%b %d, %Y %I:%M %p') }}</small>
                                </div>
                                {% if message.job %}
                                    <div class="mt-2 d-flex align-items-center gap-2 text-primary">
                                        <i class="bi bi-briefcase"></i>
                                        <a class="link-primary text-decoration-underline" href="{{ url_for('view_job', job_id=message.job.id) }}">{{ message.job.title }}</a>
                                    </div>
                                {% endif %}
                            </div>
                        {% endfor %}
                    {% else %}
                        <div class="empty-state">No messages yet. Conversations will appear here.</div>
                    {% endif %}
                </div>
            </div>
        </div>

        <link href='https://cdn.jsdelivr.net/npm/fullcalendar@6.1.10/index.global.min.css' rel='stylesheet' />
        <script src='https://cdn.jsdelivr.net/npm/fullcalendar@6.1.10/index.global.min.js'></script>

        <script>
        document.addEventListener('DOMContentLoaded', function() {
            const navLinks = document.querySelectorAll('.dashboard-tabs .nav-link');
            navLinks.forEach(link => {
                link.addEventListener('click', function(e) {
                    if (this.getAttribute('href').startsWith('#')) {
                        e.preventDefault();
                        document.querySelector(this.getAttribute('href'))?.scrollIntoView({ behavior: 'smooth' });
                        navLinks.forEach(l => l.classList.remove('active'));
                        this.classList.add('active');
                    }
                });
            });

            const calendarEl = document.getElementById('client-mini-calendar');
            const calendar = new FullCalendar.Calendar(calendarEl, {
                initialView: 'dayGridMonth',
                height: 320,
                headerToolbar: {
                    left: 'title',
                    center: '',
                    right: 'prev,next'
                },
                events: {{ events | tojson }},
                eventDisplay: 'block',
                eventDidMount: function(info) {
                    if (info.event.extendedProps.status === 'Canceled') {
                        info.el.style.textDecoration = 'line-through';
                    }
                },
                eventClick: function(info) {
                    window.location.href = "/edit_call/" + info.event.id;
                }
            });
            calendar.render();
        });
        </script>

        {% endblock %}''',




    'register_doctor.html' : '''{% extends "base.html" %}
    {% block content %}
        <h2>Create Doctor Account</h2>
        <form method="post">
            {{ form.hidden_tag() }}
            {{ form.username.label }} {{ form.username(class="form-control") }}
            {{ form.email.label }} {{ form.email(class="form-control") }}
            {{ form.password.label }} {{ form.password(class="form-control") }}
            {{ form.submit(class="btn btn-primary mt-3") }}
        </form>
        {% endblock %}''',

    'post_job.html': '''{% extends "base.html" %}
    {% block content %}
    <div class="container py-4">
        <div class="row justify-content-center">
            <div class="col-lg-8">
                <div class="card shadow-sm border-0">
                    <div class="card-body p-4">
                        <div class="d-flex align-items-center mb-3">
                            <div class="bg-primary text-white rounded-circle d-flex align-items-center justify-content-center me-3" style="width:48px; height:48px;">
                                <i class="bi bi-briefcase-fill"></i>
                            </div>
                            <div>
                                <h3 class="mb-0">Post a New Position</h3>
                                <small class="text-muted">Share the essentials so we can match you with the right clinicians.</small>
                            </div>
                        </div>

                        <form method="post" class="mt-3">
                            {{ form.hidden_tag() }}

                            <div class="mb-3">
                                <label class="form-label fw-semibold">{{ form.title.label.text }}</label>
                                {{ form.title(class="form-control form-control-lg", placeholder="e.g., Family Medicine Physician") }}
                            </div>

                            <div class="row g-3 mb-3">
                                <div class="col-md-7">
                                    <label class="form-label fw-semibold">{{ form.location.label.text }}</label>
                                    <div class="input-group">
                                        <span class="input-group-text bg-white"><i class="bi bi-geo-alt"></i></span>
                                        {{ form.location(class="form-control", placeholder="City, State") }}
                                    </div>
                                </div>
                                <div class="col-md-5">
                                    <label class="form-label fw-semibold">{{ form.salary.label.text }}</label>
                                    <div class="input-group">
                                        <span class="input-group-text bg-white"><i class="bi bi-cash-stack"></i></span>
                                        {{ form.salary(class="form-control", placeholder="Compensation details") }}
                                    </div>
                                </div>
                            </div>

                            <div class="mb-3">
                                <label class="form-label fw-semibold">{{ form.description.label.text }}</label>
                                {{ form.description(class="form-control", rows=5, placeholder="Summarize responsibilities, schedule, and ideal experience") }}
                                <div class="form-text">Provide key highlights to attract the best candidates.</div>
                            </div>

                            <div class="d-flex justify-content-end">
                                <a href="{{ url_for('home') }}" class="btn btn-outline-secondary me-2">Cancel</a>
                                {{ form.submit(class="btn btn-primary px-4") }}
                            </div>
                        </form>
                    </div>
                </div>
            </div>
        </div>
    </div>
    {% endblock %}''',

    'job_requirements.html': '''{% extends "base.html" %}
    {% block content %}
    <style>
        .wizard-shell {
            background: radial-gradient(circle at 10% 20%, rgba(99, 102, 241, 0.08), transparent 35%),
                        radial-gradient(circle at 90% 0%, rgba(16, 185, 129, 0.08), transparent 25%);
            border-radius: 20px;
            padding: 1rem 1.5rem 2rem;
            box-shadow: 0 30px 80px rgba(0,0,0,0.08);
        }
        .glass-card {
            background: rgba(255,255,255,0.9);
            border: 1px solid rgba(255,255,255,0.5);
            border-radius: 18px;
            box-shadow: 0 20px 60px rgba(31,41,55,0.12);
            backdrop-filter: blur(8px);
        }
        .section-title {
            font-size: 1.05rem;
            font-weight: 700;
            color: #0f172a;
        }
    </style>
    <div class="container py-4">
        <div class="row justify-content-center">
            <div class="col-lg-10">
                <div class="wizard-shell">
                    <div class="d-flex flex-wrap justify-content-between align-items-center mb-3">
                        <div>
                            <p class="text-uppercase text-muted small mb-1">Job Qualification Matrix</p>
                            <h2 class="fw-bold mb-0">{{ job.title }} — Specific Needs</h2>
                            <p class="text-muted mb-0">Align these requirements with the doctor profile to ensure a great match.</p>
                        </div>
                        <div class="d-flex gap-2 flex-wrap">
                            <a class="btn btn-outline-secondary" href="{{ url_for('client_dashboard') if current_user.role == 'client' else url_for('home') }}">Back to Dashboard</a>
                        </div>
                    </div>

                    <form method="post" class="glass-card p-4">
                        {{ form.hidden_tag() }}

                        <div class="row g-4">
                            <div class="col-lg-6">
                                <p class="section-title">Role & Clinical Focus</p>
                                <div class="mb-3">{{ form.position.label(class="form-label fw-semibold") }} {{ form.position(class="form-select") }}</div>
                                <div class="mb-3">{{ form.specialty.label(class="form-label fw-semibold") }} {{ form.specialty(class="form-select") }}</div>
                                <div class="mb-3">{{ form.subspecialty.label(class="form-label fw-semibold") }} {{ form.subspecialty(class="form-control") }}</div>
                                <div class="mb-3">{{ form.certification.label(class="form-label fw-semibold") }} {{ form.certification(class="form-select") }}</div>
                                <div class="mb-3">{{ form.certification_specialty_area.label(class="form-label fw-semibold") }} {{ form.certification_specialty_area(class="form-control") }}</div>
                                <div class="mb-3">{{ form.clinically_active.label(class="form-label fw-semibold") }} {{ form.clinically_active(class="form-select") }}</div>
                            </div>

                            <div class="col-lg-6">
                                <p class="section-title">Systems & Communication</p>
                                <div class="mb-3">
                                    <label class="form-label fw-semibold">{{ form.emr.label.text }}</label>
                                    <div class="border rounded p-2" style="max-height: 200px; overflow-y: auto;">
                                        {% for subfield in form.emr %}
                                            <div class="form-check">{{ subfield(class="form-check-input", id=subfield.id) }} <label class="form-check-label" for="{{ subfield.id }}">{{ subfield.label.text }}</label></div>
                                        {% endfor %}
                                    </div>
                                    <div class="mt-2">{{ form.emr_other(class="form-control", placeholder="Other EMR systems") }}</div>
                                </div>

                                <div class="mb-3">
                                    <label class="form-label fw-semibold">{{ form.languages.label.text }}</label>
                                    <div class="border rounded p-2" style="max-height: 200px; overflow-y: auto;">
                                        {% for subfield in form.languages %}
                                            <div class="form-check">{{ subfield(class="form-check-input", id=subfield.id) }} <label class="form-check-label" for="{{ subfield.id }}">{{ subfield.label.text }}</label></div>
                                        {% endfor %}
                                    </div>
                                    <div class="mt-2">{{ form.language_other(class="form-control", placeholder="Other languages") }}</div>
                                </div>
                            </div>
                        </div>

                        <div class="row g-4 mt-2">
                            <div class="col-lg-6">
                                <p class="section-title">Location Fit</p>
                                <div class="mb-3">
                                    <label class="form-label fw-semibold">{{ form.states_required.label.text }}</label>
                                    <div class="border rounded p-2" style="max-height: 200px; overflow-y: auto;">
                                        {% for subfield in form.states_required %}
                                            <div class="form-check">{{ subfield(class="form-check-input", id=subfield.id) }} <label class="form-check-label" for="{{ subfield.id }}">{{ subfield.label.text }}</label></div>
                                        {% endfor %}
                                    </div>
                                </div>
                                <div class="mb-3">
                                    <label class="form-label fw-semibold">{{ form.states_preferred.label.text }}</label>
                                    <div class="border rounded p-2" style="max-height: 200px; overflow-y: auto;">
                                        {% for subfield in form.states_preferred %}
                                            <div class="form-check">{{ subfield(class="form-check-input", id=subfield.id) }} <label class="form-check-label" for="{{ subfield.id }}">{{ subfield.label.text }}</label></div>
                                        {% endfor %}
                                    </div>
                                </div>
                            </div>

                            <div class="col-lg-6">
                                <p class="section-title">Other Needs</p>
                                <div class="mb-3 form-check">{{ form.sponsorship_supported(class="form-check-input", id="sponsorshipSupported") }} <label class="form-check-label" for="sponsorshipSupported">{{ form.sponsorship_supported.label.text }}</label></div>
                                <div class="mb-3">{{ form.salary_range.label(class="form-label fw-semibold") }} {{ form.salary_range(class="form-control", placeholder="e.g., $180k - $220k") }}</div>
                                <div class="mb-3">{{ form.notes.label(class="form-label fw-semibold") }} {{ form.notes(class="form-control", rows=4, placeholder="Add specifics about schedule, call, patient mix, procedures, etc.") }}</div>
                            </div>
                        </div>

                        <div class="d-flex justify-content-end mt-3">
                            <a class="btn btn-outline-secondary me-2" href="{{ url_for('client_dashboard') if current_user.role == 'client' else url_for('home') }}">Cancel</a>
                            {{ form.submit(class="btn btn-success px-4") }}
                        </div>
                    </form>
                </div>
            </div>
        </div>
    </div>
    {% endblock %}''',

    'register.html': '''{% extends "base.html" %}
    {% block content %}
        <h2>Register New User</h2>
        <form method="post">
            {{ form.hidden_tag() }}
            <div class="mb-3">
                {{ form.username.label }} {{ form.username(class="form-control") }}
            </div>
            <div class="mb-3">
                {{ form.password.label }} {{ form.password(class="form-control", type="password") }}
            </div>
            {{ form.submit(class="btn btn-success") }}
        </form>
        {% endblock %}''',

    'add_doctor.html': '''{% extends "base.html" %}
        {% block content %}
        <h2>Add New Doctor</h2>

        <form method="post" enctype="multipart/form-data">
            {{ form.hidden_tag() }}

            <div class="mb-3">
                {{ form.profile_picture.label }}
                {{ form.profile_picture(class="form-control", id="profileInput") }}
            </div>

            <!-- Image Preview for Cropping -->
            <div class="text-center mt-3">
                <img id="preview" style="max-width: 300px; display:none; border-radius: 50%;">
            </div>

            <!-- Hidden input to hold cropped base64 image -->
            <input type="hidden" name="cropped_image_data" id="croppedImageData">

            <!-- Button to trigger cropping -->
            <button type="button" class="btn btn-info mt-2" id="cropBtn" style="display:none;">Crop and Save</button>


            <div class="mb-3">{{ form.position.label }} {{ form.position(class="form-select") }}</div>
            <div class="mb-3">{{ form.specialty.label }} {{ form.specialty(class="form-select") }}</div>
            <div class="mb-3">{{ form.subspecialty.label }} {{ form.subspecialty(class="form-control") }}</div>
            <div class="mb-3">{{ form.first_name.label }} {{ form.first_name(class="form-control") }}</div>
            <div class="mb-3">{{ form.last_name.label }} {{ form.last_name(class="form-control") }}</div>
            <div class="mb-3">{{ form.email.label }} {{ form.email(class="form-control") }}</div>
            <div class="mb-3">{{ form.phone.label }} {{ form.phone(class="form-control") }}</div>
            <div class="mb-3">{{ form.alt_phone.label }} {{ form.alt_phone(class="form-control") }}</div>
            <div class="mb-3">
                <label class="form-label fw-semibold">{{ form.address.label.text }}</label>
                {{ form.address(class="form-control", placeholder="Street address (Line 1)") }}
            </div>
            <div class="row mb-3 g-3">
                <div class="col-md-6">
                    <label class="form-label fw-semibold">{{ form.city.label.text }}</label>
                    {{ form.city(class="form-control", placeholder="City") }}
                </div>
                <div class="col-md-6">
                    <label class="form-label fw-semibold">{{ form.state.label.text }}</label>
                    {{ form.state(class="form-select") }}
                </div>
            </div>

            <div class="position-section md-do-fields">
                <h4>MD/DO Information</h4>
                <div class="mb-3">{{ form.medical_school.label }} {{ form.medical_school(class="form-control") }}</div>
                <div class="mb-3">{{ form.med_grad_month_year.label }} {{ form.med_grad_month_year(class="form-control") }}</div>
                <div class="mb-3">{{ form.residency.label }} {{ form.residency(class="form-control") }}</div>
                <div class="mb-3">{{ form.residency_grad_month_year.label }} {{ form.residency_grad_month_year(class="form-control") }}</div>

                <h4>Fellowships</h4>
                <div class="mb-3">
                    {{ form.num_fellowships.label }} {{ form.num_fellowships(class="form-select", id="num_fellowships") }}
                </div>
                <div id="fellowship_fields">
                    {% for fellowship_field, date_field in zip(form.fellowship, form.fellowship_grad_month_year) %}
                    <div class="border p-3 mb-3 rounded fellowship-case">
                        <div class="mb-2">{{ fellowship_field.label }} {{ fellowship_field(class="form-control") }}</div>
                        <div class="mb-2">{{ date_field.label }} {{ date_field(class="form-control") }}</div>
                    </div>
                    {% endfor %}
                </div>

            </div>

            <div class="position-section np-pa-fields">
                <h4>NP/PA Information</h4>
                <div class="mb-3">{{ form.bachelors.label }} {{ form.bachelors(class="form-control") }}</div>
                <div class="mb-3">{{ form.bachelors_grad_month_year.label }} {{ form.bachelors_grad_month_year(class="form-control") }}</div>
                <div class="mb-3">{{ form.msn.label }} {{ form.msn(class="form-control") }}</div>
                <div class="mb-3">{{ form.msn_grad_month_year.label }} {{ form.msn_grad_month_year(class="form-control") }}</div>
                <div class="mb-3">{{ form.dnp.label }} {{ form.dnp(class="form-control") }}</div>
                <div class="mb-3">{{ form.dnp_grad_month_year.label }} {{ form.dnp_grad_month_year(class="form-control") }}</div>
            </div>

           <h4>Malpractice Cases</h4>
            <div class="mb-3">
                {{ form.num_malpractice_cases.label }} {{ form.num_malpractice_cases(class="form-select", id="num_malpractice_cases") }}
            </div>
            <div id="malpractice_fields">
                {% for case in form.malpractice_cases %}
                <div class="border p-3 mb-3 rounded malpractice-case">
                    <div class="mb-2">{{ case.incident_year.label }} {{ case.incident_year(class="form-control") }}</div>
                    <div class="mb-2">{{ case.outcome.label }} {{ case.outcome(class="form-select") }}</div>
                    <div class="mb-2">{{ case.payout_amount.label }} {{ case.payout_amount(class="form-control") }}</div>
                    <div class="mb-2">{{ case.case_explanation.label }} {{ case.case_explanation(class="form-control", rows=3) }}</div>
                </div>
                {% endfor %}
            </div>

            <div class="mb-3">{{ form.certification.label }} {{ form.certification(class="form-select") }}</div>
            <div class="mb-3">{{ form.certification_specialty_area.label }} {{ form.certification_specialty_area(class="form-control") }}</div>
            <div class="mb-3">{{ form.clinically_active.label }} {{ form.clinically_active(class="form-select", id="clinically_active") }}</div>
            <div class="mb-3" id="last_active_field" style="display:none;">{{ form.last_clinically_active.label }} {{ form.last_clinically_active(class="form-control") }}</div>

            <div class="position-section position-shared">
                <h5>Practice Details</h5>
                <div class="mb-3">{{ form.additional_training.label }} {{ form.additional_training(class="form-control") }}</div>
                <div class="form-check mb-3">{{ form.sponsorship_needed(class="form-check-input") }} {{ form.sponsorship_needed.label(class="form-check-label") }}</div>
            </div>
            <div class="row mb-3">
                <div class="col-md-6 mb-3 mb-md-0">
                    <label class="form-label"><strong>{{ form.emr.label }}</strong></label>
                    <div class="d-flex flex-wrap border rounded p-2" style="max-height:300px; overflow-y:auto;">
                        {% for emr_option in form.emr %}
                        <div class="form-check me-3" style="width:200px;">{{ emr_option(class="form-check-input") }} {{ emr_option.label(class="form-check-label") }}</div>
                        {% endfor %}
                    </div>
                    <div class="mt-2">{{ form.emr_other.label(class="form-label") }} {{ form.emr_other(class="form-control", placeholder="Enter other EMR systems") }}</div>
                </div>
                <div class="col-md-6">
                    <label class="form-label"><strong>{{ form.languages.label }}</strong></label>
                    <div class="d-flex flex-wrap border rounded p-2" style="max-height:300px; overflow-y:auto;">
                        {% for language in form.languages %}
                        <div class="form-check me-3" style="width:180px;">{{ language(class="form-check-input") }} {{ language.label(class="form-check-label") }}</div>
                        {% endfor %}
                    </div>
                    <div class="mt-2">{{ form.language_other.label(class="form-label") }} {{ form.language_other(class="form-control", placeholder="Enter other languages") }}</div>
                </div>
            </div>

            <div class="row mb-3">
                <div class="col-md-6">
                    <label class="form-label"><strong>{{ form.states_licensed.label }}</strong></label>
                    <div class="d-flex flex-wrap border rounded p-2" style="max-height:300px; overflow-y:auto;">
                        {% for state in form.states_licensed %}
                        <div class="form-check me-3" style="width:100px;">{{ state(class="form-check-input") }} {{ state.label(class="form-check-label") }}</div>
                        {% endfor %}
                    </div>
                </div>
                <div class="col-md-6">
                    <label class="form-label"><strong>{{ form.states_willing_to_work.label }}</strong></label>
                    <div class="d-flex flex-wrap border rounded p-2" style="max-height:300px; overflow-y:auto;">
                        {% for state in form.states_willing_to_work %}
                        <div class="form-check me-3" style="width:100px;">{{ state(class="form-check-input") }} {{ state.label(class="form-check-label") }}</div>
                        {% endfor %}
                    </div>
                </div>
            </div>

            <div class="mb-3">{{ form.salary_expectations.label }} {{ form.salary_expectations(class="form-control") }}</div>

            {{ form.submit(class="btn btn-success") }}
        </form>

        <script>
            function toggleFields(selectorId, classSelector) {
                 const sel = document.getElementById(selectorId),
                     fields = document.querySelectorAll(classSelector);
                 sel.addEventListener('change', () => {
                     const n = parseInt(sel.value, 10);
                     fields.forEach((el, i) => el.style.display = i < n ? 'block' : 'none');
                 });
                 sel.dispatchEvent(new Event('change'));
             }

             toggleFields('num_malpractice_cases', '.malpractice-case');
             toggleFields('num_fellowships', '.fellowship-case');

             const positionSelect = document.getElementById('position');
             const mdDoSections = document.querySelectorAll('.md-do-fields');
             const npPaSections = document.querySelectorAll('.np-pa-fields');

             function syncPositionSections() {
                 const pos = positionSelect.value;
                 const showMdDo = pos === 'MD' || pos === 'DO';
                 const showNpPa = pos === 'NP' || pos === 'PA';

                 mdDoSections.forEach(el => el.style.display = showMdDo ? '' : 'none');
                 npPaSections.forEach(el => el.style.display = showNpPa ? '' : 'none');
             }

             positionSelect.addEventListener('change', syncPositionSections);
             syncPositionSections();

             document.getElementById('clinically_active').addEventListener('change', function () {
                 const selectedOption = this.value;
                 const lastActiveDiv = document.getElementById('last_active_field');
                 if (selectedOption === 'No') {
                     lastActiveDiv.style.display = 'block';
                 } else {
                     lastActiveDiv.style.display = 'none';
                     document.getElementById("last_clinically_active").value = '';
                 }
             });
             document.getElementById('clinically_active').dispatchEvent(new Event('change'));
         </script>
         {% endblock %}''',

    'client_my_jobs.html': '''{% extends "base.html" %}
        {% block content %}
        <h2 class="mb-4">My Posted Jobs</h2>

        <form method="get" class="row g-3 mb-4">
            <div class="col-md-4">
                <input type="text" name="keyword" value="{{ keyword }}" class="form-control" placeholder="Enter Job Title / Keyword(s)">
            </div>
            <div class="col-md-4">
                <input type="text" name="location" value="{{ location }}" class="form-control" placeholder="Enter Location">
            </div>
            <div class="col-md-2">
                <button type="submit" class="btn btn-primary w-100">Search</button>
            </div>
            <div class="col-md-2">
                <a href="{{ url_for('client_my_jobs') }}" class="btn btn-secondary w-100">Clear</a>
            </div>
        </form>

        {% if jobs %}
            {% for job in jobs %}
            <div class="card mb-4 shadow-sm">
                <div class="card-body">
                    <h4 class="card-title">{{ job.title }}</h4>
                    <h6 class="card-subtitle text-muted mb-2">{{ job.location }}</h6>
                    <p class="card-text text-truncate" style="max-height: 5.5em; overflow: hidden;">
                        {{ job.description }}
                    </p>
                    <div class="d-flex justify-content-between align-items-center">
                        <small class="text-muted">{{ job.salary }}</small>
                        <a href="{{ url_for('edit_job', job_id=job.id) }}" class="btn btn-sm btn-warning">Edit</a>
                    </div>
                </div>
            </div>
            {% endfor %}
        {% else %}
            <p>No jobs match your criteria.</p>
        {% endif %}

        <div class="d-flex gap-2 mt-4">
            <a href="{{ url_for('post_job') }}" class="btn btn-outline-primary">Post a New Job</a>
            <a href="{{ url_for('client_dashboard') }}" class="btn btn-outline-secondary">Back to Dashboard</a>
        </div>
        {% endblock %}''',

    'schedule_call.html': '''
        {% extends "base.html" %}
        {% block content %}
            <h2>Schedule Call with Doctor</h2>
            <form method="post">
                {{ form.hidden_tag() }}

                <div class="mb-3">
                    {{ form.doctor_id.label }}
                    {{ form.doctor_id(class="form-select") }}
                </div>

                <div class="mb-3">
                    {{ form.datetime.label }}
                    {{ form.datetime(class="form-control", id="datetime") }}
                </div>

                <div class="mb-3">
                    {{ form.reason.label }}
                    {{ form.reason(class="form-control") }}
                </div>

                {{ form.submit(class="btn btn-primary") }}

                <div class="mt-3">
                    <button name="send_invite" value="yes" class="btn btn-info">Send Invite to Doctor</button>
                </div>
            </form>


            <!-- Include Select2 CSS and JS -->
            <link href="https://cdn.jsdelivr.net/npm/select2@4.1.0-rc.0/dist/css/select2.min.css" rel="stylesheet" />
            <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/select2@4.1.0-rc.0/dist/js/select2.min.js"></script>

            <script>
                $(document).ready(function() {
                    $('.form-select').select2({
                        placeholder: "Select Doctor (Name | Email | Specialty)",
                        allowClear: true,
                        width: '100%'
                    });
                });
            </script>
        {% endblock %}''',

    'doctor_profile.html': '''{% extends "base.html" %}{% block content %}
        {% set show_compare = compare_mode and job_requirement %}
        {% macro compare_line(label, value, key) -%}
            {% set result = comparison.get(key, {'matches': True, 'explanation': 'Not tied to job requirements.'}) %}
            <p class="d-flex align-items-start">
                <strong class="me-1">{{ label }}:</strong>
                <span class="flex-grow-1">{{ value if value else 'Not provided' }}</span>
                {% if show_compare %}
                    <span class="ms-2 badge {% if result.matches %}bg-success{% else %}bg-danger{% endif %}">
                        {% if result.matches %}✓{% else %}✕{% endif %}
                    </span>
                    {% if result.explanation %}
                        <small class="text-muted ms-2">{{ result.explanation }}</small>
                    {% endif %}
                {% endif %}
            </p>
        {%- endmacro %}

        <div class="d-flex justify-content-between align-items-center mb-3">
            <h2 class="mb-0">Doctor Profile: {{ doctor.first_name }} {{ doctor.last_name }}</h2>
            {% if job_requirement %}
                <div>
                    <a href="{{ url_for('doctor_profile', doctor_id=doctor.id, job_id=job_id, compare='1') }}" class="btn btn-outline-primary btn-sm {% if show_compare %}disabled{% endif %}">Compare to Job Needs</a>
                    {% if show_compare %}
                        <a href="{{ url_for('doctor_profile', doctor_id=doctor.id, job_id=job_id) }}" class="btn btn-outline-secondary btn-sm ms-2">Clear Comparison</a>
                    {% endif %}
                </div>
            {% endif %}
        </div>

        {% if job_requirement %}
            <div class="alert alert-info">
                Comparing against job: <strong>{{ associated_job.title }}</strong> ({{ associated_job.location }})
            </div>
        {% endif %}

        <div class="card shadow p-4 mb-4">
            <h5 class="card-title text-primary">Basic Information</h5>
            {{ compare_line('Healthcare Provider Type', doctor.position, 'position') }}
            {{ compare_line('Specialty', doctor.specialty, 'specialty') }}
            {{ compare_line('Subspecialty', doctor.subspecialty, 'subspecialty') }}
            {{ compare_line('Email', doctor.email, 'email') }}
            {{ compare_line('Phone', doctor.phone, 'phone') }}
            {{ compare_line('Alternative Phone', doctor.alt_phone, 'alt_phone') }}
            {{ compare_line('Address', doctor.address, 'address') }}
            {{ compare_line('City of Residence', doctor.city_of_residence, 'city_of_residence') }}
        </div>

        {% if doctor.medical_school %}
        <div class="card shadow p-4 mb-4">
            <h5 class="card-title text-primary">MD/DO Education</h5>
            {{ compare_line('Medical School', doctor.medical_school, 'medical_school') }}
            {{ compare_line('Medical School Graduation', doctor.med_grad_month_year, 'med_grad_month_year') }}
            {{ compare_line('Residency', doctor.residency, 'residency') }}
            {{ compare_line('Residency Graduation', doctor.residency_grad_month_year, 'residency_grad_month_year') }}
            {{ compare_line('Fellowships', doctor.fellowship, 'fellowship') }}
            {{ compare_line('Fellowship Graduation', doctor.fellowship_grad_month_year, 'fellowship_grad_month_year') }}
        </div>
        {% endif %}

        {% if doctor.bachelors %}
        <div class="card shadow p-4 mb-4">
            <h5 class="card-title text-primary">NP/PA Education</h5>
            {{ compare_line('Bachelors Degree', doctor.bachelors, 'bachelors') }}
            {{ compare_line('Bachelors Graduation', doctor.bachelors_grad_month_year, 'bachelors_grad_month_year') }}
            {{ compare_line('MSN', doctor.msn, 'msn') }}
            {{ compare_line('MSN Graduation', doctor.msn_grad_month_year, 'msn_grad_month_year') }}
            {{ compare_line('DNP', doctor.dnp, 'dnp') }}
            {{ compare_line('DNP Graduation', doctor.dnp_grad_month_year, 'dnp_grad_month_year') }}
            {{ compare_line('Additional Training', doctor.additional_training, 'additional_training') }}
            {{ compare_line('Sponsorship Needed', 'Yes' if doctor.sponsorship_needed else 'No', 'sponsorship') }}
        </div>
        {% endif %}

        <div class="card shadow p-4 mb-4">
            <h5 class="card-title text-primary">Licensing & Work Preferences</h5>
            {{ compare_line('Certification', doctor.certification, 'certification') }}
            {{ compare_line('Certification Specialty Area', doctor.certification_specialty_area, 'certification_specialty_area') }}
            {{ compare_line('Clinically Active', doctor.clinically_active, 'clinically_active') }}
            {{ compare_line('EMR', doctor.emr.replace(',', ', ') if doctor.emr else '', 'emr') }}
            {{ compare_line('Languages', doctor.languages, 'languages') }}
            {{ compare_line('States Licensed', doctor.states_licensed, 'states_required') }}
            {{ compare_line('States Willing to Work', doctor.states_willing_to_work, 'states_preferred') }}
            {{ compare_line('Salary Expectation (Total Compensation)', ('$' ~ ('{:.0f}'.format(doctor.salary_expectations))) if doctor.salary_expectations else 'Not provided', 'salary') }}
        </div>

        <div class="card shadow p-4 mb-4">
            <h5 class="card-title text-primary">Malpractice Cases</h5>
            {% if malpractice_cases %}
                {% for case in malpractice_cases %}
                    {{ compare_line('Incident Year', case.incident_year, 'malpractice_incident_year_' ~ loop.index) }}
                    {{ compare_line('Outcome', case.outcome, 'malpractice_outcome_' ~ loop.index) }}
                    {{ compare_line('Payout Amount', '$' ~ case.payout_amount, 'malpractice_payout_' ~ loop.index) }}
                    <hr>
                {% endfor %}
            {% else %}
                {{ compare_line('Malpractice Cases', 'No malpractice cases reported.', 'malpractice_none') }}
            {% endif %}
        </div>

        <a href="{{ url_for('edit_doctor', doctor_id=doctor.id) }}" class="btn btn-warning">Edit Profile</a>
        <a href="{{ url_for('send_job_to_doctor', doctor_id=doctor.id) }}" class="btn btn-info">Send Job Posting</a>

        {% if current_user.role in ['client', 'admin'] %}
            {% set first_job = current_user.jobs[0] if current_user.jobs|length > 0 else None %}
            {% if first_job %}
                <a href="{{ url_for('send_invite', doctor_id=doctor.id, job_id=first_job.id) }}" class="btn btn-success">Schedule Call</a>
            {% else %}
                <button class="btn btn-secondary" disabled title="Post a job first">Schedule Call</button>
            {% endif %}
        {% endif %}

        <a href="{{ url_for('doctors') }}" class="btn btn-secondary">Back to Doctors List</a>
    {% endblock %}''',

    'register_client.html': '''
        {% extends "base.html" %}
        {% block content %}
            <h2>Create Client Account</h2>
            <form method="post">
                {{ form.hidden_tag() }}
                <div class="mb-3">
                    {{ form.username.label }} {{ form.username(class="form-control") }}
                </div>
                <div class="mb-3">
                    {{ form.email.label }} {{ form.email(class="form-control") }}
                </div>
                <div class="mb-3">
                    {{ form.password.label }} {{ form.password(class="form-control") }}
                </div>
                {{ form.submit(class="btn btn-primary") }}
            </form>
        {% endblock %}''',

    'calls.html': '''{% extends "base.html" %}
        {% block content %}
            <h2>Scheduled Calls with Doctors</h2>

            <div id='calendar'></div>

            <script>
                document.addEventListener('DOMContentLoaded', function() {
                    var calendarEl = document.getElementById('calendar');

                    var calendar = new FullCalendar.Calendar(calendarEl, {
                        initialView: 'dayGridMonth',
                        headerToolbar: {
                            left: 'prev,next today',
                            center: 'title',
                            right: 'dayGridMonth,timeGridWeek,timeGridDay'
                        },
                        height: 650,
                        events: {{ events | tojson }},
                        
                        // Add this block to handle event clicks:
                        eventClick: function(info) {
                            window.location.href = "/edit_call/" + info.event.id;
                        }
                    });

                    calendar.render();
                });
            </script>
        {% endblock %}''',


'doctor_dashboard.html':'''{% extends "base.html" %}
    {% block content %}
    <style>
        .doctor-dashboard {
            color: #1f2937;
        }
        .doctor-dashboard .glass-card {
            background: linear-gradient(145deg, #f9fbff, #eef4ff);
            border: 1px solid #dbe7ff;
            box-shadow: 0 12px 28px rgba(17, 24, 39, 0.08);
            border-radius: 18px;
            overflow: hidden;
        }
        .doctor-dashboard .hero-card {
            background: radial-gradient(circle at 20% 20%, rgba(79, 70, 229, 0.08), transparent 35%),
                        radial-gradient(circle at 80% 0%, rgba(59, 130, 246, 0.16), transparent 40%),
                        linear-gradient(120deg, #f2f6ff, #e7f0ff);
            border: 1px solid #d7e7ff;
        }
        .doctor-dashboard .badge.bg-primary-soft { background: #e0edff; color: #1d4ed8; }
        .doctor-dashboard .badge.bg-secondary-soft { background: #eae9ff; color: #5b21b6; }
        .doctor-dashboard .badge.bg-dark-soft { background: #eef2ff; color: #111827; }
        .doctor-dashboard .nav-link { color: #1f2a44; border-radius: 999px; }
        .doctor-dashboard .nav-link.active, .doctor-dashboard .nav-link:hover { color: #0b3a82; background: #e0edff; }
        .inbox-item {
            padding: 12px 14px;
            background: #ffffff;
            border-radius: 12px;
            border: 1px solid #e5e7eb;
        }
        .inbox-item + .inbox-item { margin-top: 10px; }
        .inbox-item small { color: #6b7280; }
        .card-header { border-bottom: 1px solid #e5e7eb; }
        .suggested-card {
            background: #ffffff;
            border: 1px solid #d7e7ff;
            border-radius: 14px;
            padding: 16px;
            height: 100%;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .suggested-card:hover { transform: translateY(-4px); box-shadow: 0 10px 24px rgba(17, 24, 39, 0.12); }
        .suggested-card h5 { color: #0b3a82; }
        .suggested-pill { background: #e0edff; color: #0b3a82; }
        .calendar-card { min-height: 360px; }
        #mini-calendar a { color: #0b3a82; }
    </style>

    <div class="doctor-dashboard">
        <div class="glass-card hero-card d-flex flex-column flex-lg-row justify-content-between align-items-start align-items-lg-center mb-4 p-4">
            <div>
                <div class="text-uppercase small text-primary mb-2 fw-semibold">Doctor dashboard</div>
                <h2 class="fw-bold mb-2">Welcome, Dr. {{ doctor.first_name }} {{ doctor.last_name }}</h2>
                <div class="d-flex flex-wrap gap-2">
                    <span class="badge rounded-pill bg-primary-soft fw-semibold">{{ doctor.specialty or 'Specialty not set'}}</span>
                    {% if doctor.subspecialty %}
                    <span class="badge rounded-pill bg-secondary-soft fw-semibold">{{ doctor.subspecialty }}</span>
                    {% endif %}
                    {% if doctor.city_of_residence %}
                    <span class="badge rounded-pill bg-dark-soft fw-semibold"><i class="bi bi-geo-alt me-1"></i>{{ doctor.city_of_residence }}</span>
                    {% endif %}
                </div>
            </div>
            <div class="d-flex flex-wrap gap-2 mt-3 mt-lg-0">
                <a class="btn btn-outline-primary" href="{{ url_for('doctor_edit_profile') }}">Edit Profile</a>
                <a class="btn btn-outline-secondary" href="{{ url_for('doctor_inbox') }}">Open Inbox</a>
                <a class="btn btn-primary" href="{{ url_for('doctor_jobs') }}">Browse Jobs</a>
            </div>
        </div>

        <ul class="nav nav-pills mb-4 gap-2 dashboard-tabs">
            <li class="nav-item"><a class="nav-link active" href="#suggested-section">Suggested Jobs</a></li>
            <li class="nav-item"><a class="nav-link" href="#calendar-card">Calendar</a></li>
            <li class="nav-item"><a class="nav-link" href="#inbox-section">Inbox</a></li>
        </ul>

        <div class="row g-4 align-items-stretch">
            <div class="col-lg-8" id="suggested-section">
                <div class="card glass-card h-100">
                    <div class="card-header d-flex justify-content-between align-items-center px-4 py-3">
                        <div>
                            <div class="text-uppercase small text-primary fw-semibold">Opportunities</div>
                            <h5 class="mb-0">AI-suggested roles just for you</h5>
                        </div>
                        <div class="d-flex flex-wrap gap-2">
                            <button class="btn btn-outline-primary btn-sm" id="refine-suggestions">Refine with AI</button>
                        </div>
                    </div>
                    <div class="card-body p-4">
                        <div id="suggested-loading" class="d-flex align-items-center gap-3 text-muted">
                            <div class="spinner-border text-primary" role="status"></div>
                            <div>Crafting tailored matches...</div>
                        </div>
                        <div id="suggested-list" class="d-flex flex-column gap-3" style="display:none;"></div>
                        <div id="suggested-empty" class="text-muted" style="display:none;">No matches yet. Update your specialty or try refining your criteria.</div>
                    </div>
                    <div class="card-footer d-flex justify-content-between align-items-center px-4 py-3">
                        <div class="text-muted small">Showing matches for {{ doctor.specialty or 'your specialty' }}. Ranked by specialty, pay, and location fit.</div>
                    </div>
                </div>
            </div>

            <div class="col-lg-4 d-flex flex-column gap-4">
                <div class="card glass-card calendar-card" id="calendar-card">
                    <div class="card-header d-flex justify-content-between align-items-center px-4 py-3">
                        <div>
                            <div class="text-uppercase small text-primary fw-semibold">Calendar</div>
                            <h6 class="mb-0">Calls & invites</h6>
                        </div>
                        <a class="btn btn-sm btn-outline-primary" href="{{ url_for('calls') }}">Open</a>
                    </div>
                    <div class="card-body p-3">
                        <div id="mini-calendar"></div>
                    </div>
                </div>

                <div class="card glass-card" id="invites-card">
                    <div class="card-header d-flex justify-content-between align-items-center px-4 py-3">
                        <div>
                            <div class="text-uppercase small text-primary fw-semibold">Invites</div>
                            <h6 class="mb-0">Pending actions</h6>
                        </div>
                    </div>
                    <div class="card-body p-4">
                        {% if pending_invites %}
                            {% for call in pending_invites %}
                            <div class="mb-3 p-3 rounded-3 border" style="background: #ffffff; border-color: #e5e7eb !important;">
                                <div class="d-flex justify-content-between align-items-start gap-3">
                                    <div>
                                        <div class="fw-semibold">{{ call.scheduled_by.username }}</div>
                                        <div class="text-muted">{{ call.datetime.strftime('%b %d, %Y %I:%M %p') }}</div>
                                        <div class="mt-2">{{ call.reason }}</div>
                                        {% if call.job %}
                                        <div class="mt-2 text-primary"><i class="bi bi-briefcase me-1"></i>{{ call.job.title }}</div>
                                        {% endif %}
                                    </div>
                                </div>
                                <form class="mt-3 d-flex gap-2" method="post" action="{{ url_for('handle_invite', call_id=call.id) }}">
                                    <button name="action" value="accept" class="btn btn-success btn-sm">Accept</button>
                                    <button name="action" value="decline" class="btn btn-outline-secondary btn-sm">Decline</button>
                                </form>
                            </div>
                            {% endfor %}
                        {% else %}
                            <div class="text-muted">No pending invites right now.</div>
                        {% endif %}
                    </div>
                </div>
            </div>
        </div>

        <div class="card glass-card mt-4" id="inbox-section">
            <div class="card-header d-flex justify-content-between align-items-center px-4 py-3">
                <div>
                    <div class="text-uppercase small text-primary fw-semibold">Inbox</div>
                    <h5 class="mb-0">Latest conversations</h5>
                </div>
                <a class="btn btn-sm btn-outline-primary" href="{{ url_for('doctor_inbox') }}">View all</a>
            </div>
            <div class="card-body p-4">
                {% if inbox_preview %}
                    {% for message in inbox_preview %}
                    <div class="inbox-item">
                        <div class="d-flex justify-content-between align-items-start gap-3">
                            <div>
                                <div class="fw-semibold">{{ message.sender.username if message.sender else 'System'}}</div>
                                <div class="text-muted">{{ message.content[:180] }}{% if message.content|length > 180 %}...{% endif %}</div>
                            </div>
                            <small>{{ message.timestamp.strftime('%b %d, %Y %I:%M %p') }}</small>
                        </div>
                        {% if message.job %}
                        <div class="mt-2 d-flex align-items-center gap-2 text-primary">
                            <i class="bi bi-briefcase"></i>
                            <a class="link-primary text-decoration-underline" href="{{ url_for('view_job', job_id=message.job.id) }}">{{ message.job.title }}</a>
                        </div>
                        {% endif %}
                    </div>
                    {% endfor %}
                {% else %}
                    <div class="text-muted">No messages yet. Keep an eye out for client conversations.</div>
                {% endif %}
            </div>
        </div>
    </div>

    <link href='https://cdn.jsdelivr.net/npm/fullcalendar@6.1.10/index.global.min.css' rel='stylesheet' />
    <script src='https://cdn.jsdelivr.net/npm/fullcalendar@6.1.10/index.global.min.js'></script>

    <script>
    document.addEventListener('DOMContentLoaded', function() {
        const navLinks = document.querySelectorAll('.dashboard-tabs .nav-link');
        navLinks.forEach(link => {
            link.addEventListener('click', function(e) {
                if (this.getAttribute('href').startsWith('#')) {
                    e.preventDefault();
                    document.querySelector(this.getAttribute('href'))?.scrollIntoView({ behavior: 'smooth' });
                    navLinks.forEach(l => l.classList.remove('active'));
                    this.classList.add('active');
                }
            });
        });

        // Calendar
        const calendarEl = document.getElementById('mini-calendar');
        const calendar = new FullCalendar.Calendar(calendarEl, {
            initialView: 'dayGridMonth',
            height: 320,
            headerToolbar: {
                left: 'title',
                center: '',
                right: 'prev,next'
            },
            events: {{ events | tojson }},
            eventDisplay: 'block',
            eventDidMount: function(info) {
                if (info.event.extendedProps.status === 'Canceled') {
                    info.el.style.textDecoration = 'line-through';
                }
            },
            eventClick: function(info) {
                window.location.href = "/doctor/call/" + info.event.id;
            }
        });
        calendar.render();

        // Suggested jobs via AI
        const list = document.getElementById('suggested-list');
        const loading = document.getElementById('suggested-loading');
        const emptyState = document.getElementById('suggested-empty');
        const refineBtn = document.getElementById('refine-suggestions');
        const refineModalEl = document.getElementById('refineModal');
        const refineForm = document.getElementById('refineForm');
        const refineNotes = document.getElementById('refineNotes');
        const refineStatus = document.getElementById('refineStatus');
        const modal = new bootstrap.Modal(refineModalEl);
        const cacheKey = `doctorSuggestedTop3_{{ doctor.id }}`;
        const baseKey = `doctorSuggestedBase_{{ doctor.id }}`;
        const metaKey = `doctorSuggestedMeta_{{ doctor.id }}`;
        const profileSignature = `{{ doctor.specialty or '' }}|{{ doctor.subspecialty or '' }}|{{ doctor.city_of_residence or '' }}|{{ doctor.states_licensed or '' }}|{{ doctor.states_willing_to_work or '' }}|{{ doctor.salary_expectations or '' }}`;

        let suggestions = [];
        let baseSuggestions = [];

        const showLoading = () => {
            loading.classList.remove('d-none');
            loading.style.display = '';
        };

        const hideLoading = () => {
            loading.classList.add('d-none');
            loading.style.display = 'none';
        };
        const escapeHtml = (str) => {
            if (!str) return '';
            return str
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;')
                .replace(/"/g, '&quot;')
                .replace(/'/g, '&#39;');
        };

        const renderSuggestions = () => {
            list.innerHTML = '';
            suggestions.slice(0, 3).forEach((job) => {
                const row = document.createElement('div');
                row.className = 'suggested-card d-flex flex-column flex-lg-row align-items-start align-items-lg-center gap-3';
                row.innerHTML = `
                    <div class="flex-grow-1">
                        <div class="d-flex justify-content-between flex-wrap align-items-start gap-3">
                            <div>
                                <h5 class="mb-1">${escapeHtml(job.title)}</h5>
                                <div class="text-muted">${escapeHtml(job.location || 'Location TBD')}</div>
                                ${job.salary ? `<div class="mt-2 text-primary fw-semibold">${escapeHtml(job.salary)}</div>` : ''}
                            </div>
                            <span class="badge suggested-pill align-self-start">Score: ${Math.round(job.score || 0)}</span>
                        </div>
                        <p class="mt-3 mb-0 text-secondary">${escapeHtml(job.rationale)}</p>
                    </div>
                    <a class="btn btn-outline-primary btn-sm align-self-stretch" href="/doctor/job/${job.id}">View role</a>
                `;
                list.appendChild(row);
            });
        };

        const saveCache = (topThree, baseList) => {
            localStorage.setItem(cacheKey, JSON.stringify(topThree));
            localStorage.setItem(baseKey, JSON.stringify(baseList));
            localStorage.setItem(metaKey, profileSignature);
        };
        const showSuggestions = () => {
            hideLoading();
            if (suggestions.length === 0) {
                emptyState.style.display = '';
                list.style.display = 'none';
                return;
            }
            emptyState.style.display = 'none';
            list.style.display = '';
            renderSuggestions();
        };

        const fetchSuggestions = () => {
            showLoading();
            list.style.display = 'none';
            emptyState.style.display = 'none';
            fetch('{{ url_for('doctor_suggested_jobs') }}')
                .then(res => res.json())
                .then(data => {
                    baseSuggestions = data.suggestions || [];
                    suggestions = baseSuggestions.slice(0, 3);
                    saveCache(suggestions, baseSuggestions);
                    showSuggestions();
                })
                .catch(() => {
                    emptyState.style.display = '';
                    emptyState.textContent = 'Unable to fetch suggestions right now.';
                })
                .finally(() => {
                    hideLoading();
                });
        };

        const cachedTop = localStorage.getItem(cacheKey);
        const cachedBase = localStorage.getItem(baseKey);
        const cachedMeta = localStorage.getItem(metaKey);
        const profileChanged = cachedMeta !== profileSignature;

        if (cachedTop && cachedBase && !profileChanged) {
            try {
                suggestions = JSON.parse(cachedTop) || [];
                baseSuggestions = JSON.parse(cachedBase) || [];
                showSuggestions();
            } catch (e) {
                fetchSuggestions();
            }
        } else {
            fetchSuggestions();
        }

        refineBtn.addEventListener('click', () => {
            refineNotes.value = '';
            refineStatus.textContent = '';
            modal.show();
        });

        refineForm.addEventListener('submit', (e) => {
            e.preventDefault();
            const context = refineNotes.value.trim();
            refineStatus.textContent = 'Re-ranking with your preferences...';
            fetch('{{ url_for('doctor_refine_suggestions') }}', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    context,
                    jobs: baseSuggestions,
                })
            })
            .then(res => res.json())
            .then(data => {
                suggestions = (data.suggestions || []).slice(0, 3);
                if ((data.base || []).length) {
                    baseSuggestions = data.base;
                }
                saveCache(suggestions, baseSuggestions);
                showSuggestions();
                refineStatus.textContent = 'Updated matches saved for this session.';
            })
            .catch(() => {
                refineStatus.textContent = 'Unable to refine right now. Please try again soon.';
            })
            .finally(() => {
                modal.hide();
            });
        });
    });
    </script>

    <!-- Refine Suggestions Modal -->
    <div class="modal fade" id="refineModal" tabindex="-1" aria-labelledby="refineModalLabel" aria-hidden="true">
      <div class="modal-dialog modal-lg modal-dialog-centered">
        <div class="modal-content" style="border-radius:20px;">
          <div class="modal-header">
            <h5 class="modal-title" id="refineModalLabel">Narrow these matches</h5>
            <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
          </div>
          <form id="refineForm">
            <div class="modal-body">
              <p class="text-muted mb-3">Tell us more (e.g., schedule needs, call expectations, rural/urban preferences). We'll re-rank the current matches without inventing new jobs.</p>
              <textarea class="form-control" id="refineNotes" rows="3" placeholder="Add extra details to refine your top matches"></textarea>
              <div class="mt-2 text-muted small">Your top three matches will be saved for this session even after you navigate away.</div>
            </div>
            <div class="modal-footer">
              <span class="me-auto text-success" id="refineStatus"></span>
              <button type="submit" class="btn btn-primary">Update matches</button>
            </div>
          </form>
        </div>
      </div>
    </div>
    {% endblock %}''',




    'edit_job.html': '''{% extends "base.html" %}
    {% block content %}
        <h2>Edit Job: {{ job.title }}</h2>
        <form method="post">
            {{ form.hidden_tag() }}
            <div class="mb-3">
                {{ form.title.label }} {{ form.title(class="form-control") }}
            </div>
            <div class="mb-3">
                {{ form.location.label }} {{ form.location(class="form-control") }}
            </div>
            <div class="mb-3">
                {{ form.salary.label }} {{ form.salary(class="form-control") }}
            </div>
            <div class="mb-3">
                {{ form.description.label }} {{ form.description(class="form-control") }}
            </div>
            {{ form.submit(class="btn btn-success") }}
        </form>
    {% endblock %}''',

    'doctor_jobs.html': '''{% extends "base.html" %}
{% block content %}

    <style>
        .job-card-title {
            font-weight: 700;
            color: #0b3b65;
        }

        .job-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            font-size: 0.95rem;
            color: #6c757d;
        }

        .job-meta .badge {
            background: #eef5ff;
            border: 1px solid #d6e6ff;
        }

        .job-description {
            color: #4b5563;
            display: -webkit-box;
            -webkit-line-clamp: 3;
            -webkit-box-orient: vertical;
            overflow: hidden;
        }
    </style>

    <div class="d-flex justify-content-between align-items-center mb-4">
        <h2 class="mb-0">Find Jobs</h2>
        <button class="btn btn-lg btn-info" id="aiSearchBtn" type="button">
            <i class="bi bi-stars"></i> AI Search
        </button>
    </div>

    <div id="job-map" style="width:100%;height:420px;border-radius:12px;margin-bottom:36px;"></div>

    <!-- AI Search Modal -->
    <div class="modal fade" id="aiSearchModal" tabindex="-1" aria-labelledby="aiSearchModalLabel" aria-hidden="true">
      <div class="modal-dialog modal-lg modal-dialog-centered">
        <div class="modal-content" style="border-radius:20px;">
          <div class="modal-header">
            <h5 class="modal-title" id="aiSearchModalLabel">Find Your Best Job Match (AI Powered)</h5>
            <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
          </div>
          <form id="aiSearchForm">
            <div class="modal-body">
              <div class="mb-3">
                <label><b>Lifestyle Description</b></label>
                <textarea class="form-control" name="lifestyle" rows="2"
                          placeholder="e.g. Quiet suburb, outdoor activities, work/life balance..."></textarea>
              </div>
              <div class="mb-3">
                <label><b>Job-specific Wants</b></label>
                <textarea class="form-control" name="wants" rows="2"
                          placeholder="e.g. Research, teaching, high salary, certain procedures..."></textarea>
              </div>
              <div class="mb-3">
                <label><b>Location Preferences</b></label>
                <input class="form-control" name="location"
                       placeholder="e.g. Miami, Florida, Northeast, rural, etc.">
              </div>
            </div>
            <div class="modal-footer">
              <button type="submit" class="btn btn-success">Search with AI</button>
            </div>
          </form>
          <div id="aiResultsContainer" class="p-4" style="display:none;"></div>
        </div>
      </div>
    </div>

    <form method="get" class="row g-3 mb-4">
        <div class="col-md-4">
            <input type="text" name="keyword" value="{{ keyword }}" class="form-control"
                   placeholder="Enter Job Title / Keyword(s)">
        </div>
        <div class="col-md-4">
            <input type="text" name="location" value="{{ location }}" class="form-control"
                   placeholder="Enter Location">
        </div>
        <div class="col-md-2">
            <button type="submit" class="btn btn-primary w-100">Search</button>
        </div>
        <div class="col-md-2">
            <a href="{{ url_for('doctor_jobs') }}" class="btn btn-secondary w-100">Clear</a>
        </div>
    </form>

    {% if jobs %}
        {% for job in jobs %}
        <div class="card mb-4 shadow-sm" id="job-{{ job.id }}">
            <div class="card-body">
                <div class="d-flex justify-content-between align-items-start gap-2 flex-wrap">
                    <h4 class="card-title job-card-title mb-1">{{ job.title }}</h4>
                    <a href="{{ url_for('view_job', job_id=job.id) }}"
                       class="btn btn-sm btn-outline-primary">View Job</a>
                </div>
                <div class="job-meta mb-2">
                    <span class="d-flex align-items-center">
                        <i class="bi bi-geo-alt me-1"></i>
                        {{ job.location or 'Location not provided' }}
                    </span>
                    {% if job.salary %}
                        <span class="badge rounded-pill text-primary">{{ job.salary }}</span>
                    {% endif %}
                </div>
                <p class="job-description">{{ job.description }}</p>
            </div>
        </div>
        {% endfor %}
    {% else %}
        <p>No jobs match your criteria.</p>
    {% endif %}

    <div class="d-flex gap-2 mt-4">
        <a href="{{ url_for('doctor_jobs') }}" class="btn btn-outline-primary">← Back to Full Job Board</a>
        <a href="{{ url_for('doctor_dashboard') }}" class="btn btn-outline-secondary">Back to Dashboard</a>
    </div>

    <!-- Leaflet + Bootstrap Icons + JS -->
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <link rel="stylesheet"
          href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.1/font/bootstrap-icons.css">
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>

    <script>
    const jobMarkers = {{ job_markers|tojson|safe }};

    const map = L.map('job-map').setView([37.5, -96], 4);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: 'Map &copy; OpenStreetMap contributors'
    }).addTo(map);

    let markerGroup = L.featureGroup();
    jobMarkers.forEach(markerData => {
        if (markerData.lat && markerData.lng) {
            let count = markerData.jobs.length;

            let iconHtml = `
                <div class="leaflet-marker-icon-numbered">
                    <img class="pin-img"
                         src="https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png">
                    ${count > 1 ? `<div class="marker-badge">${count}</div>` : ""}
                </div>
            `;

            let icon = L.divIcon({
                html: iconHtml,
                className: '',
                iconSize: [38, 50],
                iconAnchor: [19, 50]
            });

            let popupHTML = `
                <div class="custom-popup">
                    <div class="custom-popup-header">
                        ${markerData.jobs.length === 1
                          ? markerData.jobs[0].title
                          : `${markerData.jobs.length} jobs at this location`}
                    </div>
                    <div class="custom-job-list">
            `;

            markerData.jobs.forEach(job => {
                popupHTML += `
                    <div class="custom-job">
                        <div class="custom-job-title">${job.title}</div>
                        <a href="/doctor/job/${job.id}" target="_blank"
                           class="custom-view-job">View Job</a>
                    </div>
                `;
            });

            popupHTML += `</div></div>`;

            const marker = L.marker([markerData.lat, markerData.lng], {icon: icon}).addTo(markerGroup);
            marker.bindPopup(popupHTML);
        }
    });

    markerGroup.addTo(map);

    if (jobMarkers.length > 0) {
        try {
            map.fitBounds(markerGroup.getBounds().pad(0.2));
        } catch (e) {}
    }

    document.getElementById('aiSearchBtn').addEventListener('click', function() {
        new bootstrap.Modal(document.getElementById('aiSearchModal')).show();
        document.getElementById('aiResultsContainer').style.display = 'none';
        document.getElementById('aiResultsContainer').innerHTML = '';
    });

    document.getElementById('aiSearchForm').addEventListener('submit', function(e) {
        e.preventDefault();
        const btn = this.querySelector('button[type=submit]');
        btn.disabled = true; btn.innerText = "Searching...";
        const formData = new FormData(this);
        fetch('{{ url_for("doctor_ai_search_jobs") }}', {
            method: 'POST',
            body: formData
        }).then(res => res.json()).then(data => {
            btn.disabled = false; btn.innerText = "Search with AI";
            document.getElementById('aiResultsContainer').style.display = '';
            document.getElementById('aiResultsContainer').innerHTML = data['html'];
        }).catch(() => {
            btn.disabled = false; btn.innerText = "Search with AI";
            document.getElementById('aiResultsContainer').style.display = '';
            document.getElementById('aiResultsContainer').innerHTML = '<div class="alert alert-danger">AI search failed. Try again in a few seconds.</div>';
        });
    });
    </script>
    {% endblock %}''',
    
    'landing_page.html': '''
        {% extends "base.html" %}
        {% block content %}
        <!-- AOS Animate On Scroll CSS -->
        <link href="https://cdnjs.cloudflare.com/ajax/libs/aos/2.3.4/aos.css" rel="stylesheet" />

        <style>
            .chart-container {
                transition: transform 0.3s ease;
            }
            .chart-container:hover {
                transform: scale(1.05);
            }

            .emoji-grid {
                gap: 10px;
            }

            .emoji-table td {
                font-size: 1.5rem;
                padding: 3px;
                text-align: center;
            }

            :root {
                --brand-teal: #8ecad4;
                --brand-teal-dark: #5aa4b3;
                --brand-teal-soft: #e8f5f7;
                --brand-charcoal: #1f2b2f;
            }

            .hero {
                background: linear-gradient(180deg, #f6fbfc 0%, var(--brand-teal-soft) 100%);
                color: var(--brand-charcoal);
                padding-top: 80px;
                padding-bottom: 40px;
            }

            .highlight {
                color: var(--brand-teal-dark);
            }

            .btn-custom {
                background-color: var(--brand-teal-dark);
                color: #fff;
                border: none;
                padding: 12px 26px;
                border-radius: 40px;
                box-shadow: 0 10px 30px rgba(90, 164, 179, 0.28);
                transition: transform 0.2s ease, box-shadow 0.2s ease, background-color 0.2s ease;
            }

            .btn-custom:hover {
                background-color: var(--brand-teal);
                transform: translateY(-2px);
                box-shadow: 0 14px 36px rgba(90, 164, 179, 0.35);
            }

            .plus-sign {
                font-size: 3rem;
                color: var(--brand-teal-dark);
                margin-left: 15px;
            }

            .section-alt h5,
            .section-alt p,
            .section-alt small {
                margin: 0;
            }

            @media (max-width: 768px) {
                .plus-sign {
                margin-left: 0;
                margin-top: 10px;
                }
            }
            .emoji-table td {
                transition: transform 0.4s cubic-bezier(.3,1.5,.5,1);
                will-change: transform;
            }
            </style>


        <!-- Hero Section -->
        <div class="hero text-center">
            <div class="container" style="padding-bottom: 200px;">  
                <!-- was 60px-->
                 <img id="hero-logo" src="{{ url_for('static', filename='jobsdirectmedicalcutright.png') }}" alt="DocIV Logo" class="img-fluid mb-4" style="max-width: 500px; transition: transform 0.3s ease;">
                <p class="lead">A direct line from the <span class="highlight">Doctor</span> to the <span class="highlight">Hospital</span>, eliminating annoying recruiters forever.</p>
                <div class="mt-4">
                    <a href="{{ url_for('login') }}" class="btn btn-custom">Physician Login</a>
                    <a href="{{ url_for('login') }}" class="btn btn-custom">Hospital Login</a>
                </div>
            </div>
        </div>

        <!-- Problem Statement -->
          <div class="section text-white" style="background-color: var(--brand-teal-dark); text-align: left; padding: 30px 0;">
            <div class="container">
                <h2 class="mb-4" style="margin-left: 20px;">Cut Out the Noise</h2>
                <p style="max-width: 800px; margin-left: 20px;">
                    Did you know that the average physician is contacted by over 
                    <strong>20 recruiters per month</strong>?<br>
                    <span style="display: inline-block; margin-top: 10px;">
                        DocIV puts an end to this disruption by promising you that your precious data will never be given to any recruiters. 
                        Your profile will only be viewed by verified hospital-employed recruiters.
                    </span>
                </p>
            </div>
        </div>

        <div class="section section-alt text-center" style="padding: 50px 0;">
        <!-- was  100px-->
        <div class="container">
            <h2 class="mb-5">Why DocIV Matters</h2>
            <div class="row justify-content-center g-5 align-items-end">

            <!-- Chart 1: Horizontal Bar -->
            <div class="col-md-4 d-flex flex-column align-items-center">
                <div class="chart-container mb-4">
                <canvas id="chart-overwhelm" width="300" height="180"></canvas>
                </div>
                <h5 class="mt-2">Physician Burnout from Recruiters</h5>
                <p class="fw-bold mb-1">73% feel overwhelmed</p>
                <small class="text-muted">Source: Merritt Hawkins Survey, 2021</small>
            </div>

            <!-- Chart 2: Grid + Plus -->
            <div class="col-md-4 d-flex flex-column align-items-center">
                <div class="emoji-grid mb-4 d-flex align-items-center">
                    <table class="emoji-table" id="emojiTable">
                        {% for i in range(5) %}
                        <tr>
                        {% for j in range(10) %}
                        <td>{% if (i + j) % 2 == 0 %}📧{% else %}📱{% endif %}</td>
                        {% endfor %}
                        </tr>
                        {% endfor %}
                    </table>
                    <div class="plus-sign">+</div>
                </div>
                <h5 class="mt-2">Unwanted Contact Monthly</h5>
                <p class="fw-bold mb-1">50+ cold calls/emails</p>
                <small class="text-muted">Source: Doximity Physician Insights, 2022</small>
            </div>

            <!-- Chart 3: Pie -->
            <div class="col-md-4 d-flex flex-column align-items-center">
                <div class="chart-container mb-4" style="max-width: 230px;">
                <canvas id="chart-direct" width="230" height="230"></canvas>
                </div>
                <h5 class="mt-2">Hospitals Want Direct Communication</h5>
                <p class="fw-bold mb-1">90% prefer to skip agencies</p>
                <small class="text-muted">Source: MGMA Stat Poll, 2023</small>
            </div>

            </div>
        </div>
        </div>

        <!-- How It Works Section -->
        <div class="section text-center">
            <div class="container" style="padding-bottom: 200px;">
            <!-- was  60px-->
                <img src="{{ url_for('static', filename='dociv_workflow.png') }}" alt="DocIV Workflow Graphic" class="img-fluid mt-4 shadow rounded" style="max-width: 500px;">
            </div>
        </div>

        <!-- Call to Action -->
        <div class="section section-alt text-center">
            <div class="container">
                <h2>Ready to Experience DocIV?</h2>
                <a href="{{ url_for('login') }}" class="btn btn-custom mt-3">Get Started</a>
            </div>
        </div>

        <!-- Scripts -->
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/aos/2.3.4/aos.js"></script>
        <script>
            AOS.init({ duration: 800, once: true });

            // Mousemove parallax effect
            document.addEventListener("mousemove", function (e) {
                const logo = document.getElementById("hero-logo");
                if (!logo) return;
                const moveX = (e.clientX / window.innerWidth - 0.5) * 10;
                const moveY = (e.clientY / window.innerHeight - 0.5) * 10;
                logo.style.transform = `translate(${moveX}px, ${moveY}px)`;
            });

            // Horizontal bar chart
            new Chart(document.getElementById('chart-overwhelm'), {
                type: 'bar',
                data: {
                    labels: ['Feel overwhelmed by recruiters'],
                    datasets: [{
                        data: [73],
                        backgroundColor:  '#5aa4b3'
                    }]
                },
                options: {
                    indexAxis: 'y',
                    scales: {
                        x: {
                            min: 0,
                            max: 100,
                            ticks: {
                                stepSize: 25,
                                callback: val => val + '%'
                            }
                        },
                        y: {
                            display: false
                        }
                    },
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            callbacks: {
                                label: ctx => ctx.raw + '% of physicians'
                            }
                        }
                    }
                }
            });

            // Pie chart
            new Chart(document.getElementById('chart-direct'), {
                type: 'pie',
                data: {
                    labels: ['Prefer Direct', 'Prefer Recruiters'],
                    datasets: [{
                        data: [90, 10],
                        backgroundColor: ['#5aa4b3', '#d4e9ed']
                    }]
                },
                options: {
                    plugins: {
                        legend: { position: 'bottom' },
                        tooltip: {
                            callbacks: {
                                label: ctx => `${ctx.label}: ${ctx.raw}%`
                            }
                        }
                    }
                }
            });
            (function() {
            const table = document.getElementById('emojiTable');
            if (!table) return;

            const rows = Array.from(table.rows);
            const cols = rows[0] ? rows[0].cells.length : 0;

            // Parent for the grid (should be .emoji-grid or similar)
            const gridParent = table.parentElement;

            function handleMove(e) {
                // Get mouse X,Y relative to gridParent
                let rect = gridParent.getBoundingClientRect();
                let mx = (e.clientX - rect.left) / rect.width;
                let my = (e.clientY - rect.top) / rect.height;

                rows.forEach((row, i) => {
                    Array.from(row.cells).forEach((cell, j) => {
                        let dx = (j / (cols-1)) - mx;
                        let dy = (i / (rows.length-1)) - my;
                        let dist = Math.sqrt(dx*dx + dy*dy);

                        let angle = Math.atan2(dy, dx);
                        let mag = Math.max(0, 0.22 - dist); // lower = more local effect

                        let tx = Math.cos(angle) * mag * 150;
                        let ty = Math.sin(angle) * mag * 90;

                        let wave = Math.sin(Date.now()/350 + i*0.7 + j*0.5) * 2;

                        cell.style.transform = `translate(${tx}px, ${ty + wave}px) scale(1.12)`;
                    });
                });
            }

            function resetGrid() {
                rows.forEach(row => {
                    Array.from(row.cells).forEach(cell => {
                        cell.style.transform = '';
                    });
                });
            }

            gridParent.addEventListener('mousemove', handleMove);
            gridParent.addEventListener('mouseleave', resetGrid);
        })();
        </script>
        {% endblock %}
        ''',

    'admin_analytics.html': '''{% extends "base.html" %}
        {% block content %}
        <h2 class="mb-4">Admin Job Post Analytics</h2>

        <input class="form-control mb-3" type="text" id="jobSearch" placeholder="Search by job title, location, or description...">

        <div id="jobList">
            {% for job in job_data %}
            <div class="card mb-4 shadow-sm job-card">
                <div class="card-body">
                    <h5 class="card-title">{{ job.title }}</h5>
                    <p class="mb-0"><strong>Posted By:</strong> {{ job.client_name }} ({{ job.client_email }})</p>
                    <h6 class="card-subtitle mb-2 text-muted">{{ job.location }}</h6>
                    <p class="card-text text-truncate">{{ job.description }}</p>

                    <p><strong>Interest Count:</strong> {{ job.interest_count }}</p>

                    {% if job.interest_count > 0 %}
                        <a href="{{ url_for('download_job_applicants', job_id=job.id) }}" class="btn btn-sm btn-outline-success mb-2">
                            <i class="bi bi-file-earmark-arrow-down"></i> Download Excel
                        </a>
                        <button class="btn btn-sm btn-outline-info mb-2" data-bs-toggle="collapse" data-bs-target="#doctors-{{ job.id }}">
                            View Interested Doctors
                        </button>
                        <div class="collapse" id="doctors-{{ job.id }}">
                            <ul class="list-group">
                                {% for doc in job.interested_doctors %}
                                <li class="list-group-item d-flex justify-content-between align-items-center">
                                    <span>
                                        {{ doc.name }} – {{ doc.email }}
                                    </span>
                                    <span>
                                        {% if doc.id %}
                                            <a href="{{ url_for('doctor_profile', doctor_id=doc.id) }}"
                                            class="btn btn-sm btn-outline-primary me-2">View Profile</a>
                                            <a href="{{ url_for('send_invite', doctor_id=doc.id, job_id=job.id) }}"
                                            class="btn btn-sm btn-success">Schedule Call</a>
                                        {% else %}
                                            <span class="text-muted">No profile available</span>
                                        {% endif %}
                                    </span>
                                </li>
                                {% endfor %}
                            </ul>
                        </div>
                        <canvas id="chart-{{ job.id }}" height="100"></canvas>
                        <script>
                            const ctx{{ job.id }} = document.getElementById('chart-{{ job.id }}').getContext('2d');
                            new Chart(ctx{{ job.id }}, {
                                type: 'bar',
                                data: {
                                    labels: {{ job.interest_by_day.keys() | list | tojson }},
                                    datasets: [{
                                        label: 'Interest Count by Day',
                                        data: {{ job.interest_by_day.values() | list | tojson }},
                                        backgroundColor: 'rgba(0, 123, 255, 0.5)',
                                        borderColor: 'rgba(0, 123, 255, 1)',
                                        borderWidth: 1
                                    }]
                                },
                                options: {
                                    scales: {
                                        x: { title: { display: true, text: 'Date' } },
                                        y: { beginAtZero: true, title: { display: true, text: 'Interests' } }
                                    }
                                }
                            });
                        </script>
                    {% else %}
                        <p class="text-muted">No interest yet.</p>
                    {% endif %}
                </div>
            </div>
            {% endfor %}
        </div>

        <a href="{{ url_for('admin_dashboard') }}" class="btn btn-secondary mt-4">Back to Dashboard</a>

        <!-- Chart.js + live search -->
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script>
        document.getElementById('jobSearch').addEventListener('keyup', function () {
            const query = this.value.toLowerCase();
            document.querySelectorAll('.job-card').forEach(card => {
                const text = card.innerText.toLowerCase();
                card.style.display = text.includes(query) ? '' : 'none';
            });
        });
        
        </script>
        {% endblock %}''',

    'doctor_edit_profile.html': '''{% extends "base.html" %}
        {% block content %}
        <style>
            .wizard-shell {
                background: radial-gradient(circle at 10% 20%, rgba(99, 102, 241, 0.12), transparent 35%),
                            radial-gradient(circle at 90% 0%, rgba(16, 185, 129, 0.12), transparent 25%);
                border-radius: 20px;
                padding: 1rem 1.5rem 2rem;
                box-shadow: 0 30px 80px rgba(0,0,0,0.08);
            }
            .glass-card {
                background: rgba(255,255,255,0.82);
                border: 1px solid rgba(255,255,255,0.5);
                border-radius: 18px;
                box-shadow: 0 20px 60px rgba(31,41,55,0.1);
                backdrop-filter: blur(8px);
            }
            .step-chip {
                padding: 0.45rem 0.9rem;
                border-radius: 999px;
                background: #f4f4f5;
                color: #6b7280;
                font-weight: 600;
                border: 1px solid transparent;
                transition: all 0.2s ease;
            }
            .step-chip.active {
                background: linear-gradient(120deg, #6366f1, #22c55e);
                color: #fff;
                box-shadow: 0 10px 30px rgba(99,102,241,0.25);
            }
            .wizard-step { display: none; }
            .wizard-step.active { display: block; }
            .section-title {
                font-size: 1.05rem;
                font-weight: 700;
                color: #0f172a;
            }
        </style>

        <div class="wizard-shell">
            <div class="d-flex flex-wrap justify-content-between align-items-center gap-3 mb-3">
                <div>
                    <p class="text-uppercase text-muted small mb-1">Profile Experience</p>
                    <h2 class="fw-bold mb-0">Edit Doctor: {{ doctor.first_name }} {{ doctor.last_name }}</h2>
                </div>
                <div class="d-flex gap-2 flex-wrap">
                    <span class="step-chip active" data-step="0">Profile</span>
                    <span class="step-chip" data-step="1">Education</span>
                    <span class="step-chip" data-step="2">Practice</span>
                    <span class="step-chip" data-step="3">Preferences</span>
                </div>
            </div>

            <div class="progress mb-4" style="height: 10px;">
                <div class="progress-bar bg-success" id="progressBar" style="width: 25%;"></div>
            </div>

            <form method="post" enctype="multipart/form-data" id="profileWizard" class="glass-card p-4">
                {{ form.hidden_tag() }}

                <div class="wizard-step active" data-step="0">
                    <div class="row g-4 align-items-start">
                        <div class="col-md-4">
                            {% if doctor.profile_picture %}
                            <div class="mb-3 text-center">
                                <p class="fw-semibold mb-2">Current Photo</p>
                                <img src="{{ url_for('static', filename=doctor.profile_picture) }}" class="img-thumbnail shadow-sm rounded-circle" style="max-width: 180px;">
                            </div>
                            {% endif %}

                            <div class="mb-3">
                                <label class="form-label fw-semibold">{{ form.profile_picture.label.text }}</label>
                                {{ form.profile_picture(class="form-control", id="profileInput") }}
                                <small class="text-muted">Upload a clear headshot to refresh your profile.</small>
                            </div>

                            <div class="text-center mt-3" id="crop-container" style="display:none;">
                                <div style="display:inline-block; width:260px; height:260px; border-radius:50%; overflow:hidden; background:#f0f0f0; position:relative;">
                                    <img id="preview" style="position:absolute; top:0; left:0; min-width:100%; min-height:100%;">
                                </div>
                            </div>

                            <input type="hidden" name="cropped_image_data" id="croppedImageData">
                            <button type="button" class="btn btn-outline-primary w-100 mt-2" id="cropBtn" style="display:none;">Crop and Save</button>
                        </div>

                        <div class="col-md-8">
                            <div class="row g-3">
                                <div class="col-md-6">{{ form.position.label(class="form-label fw-semibold") }} {{ form.position(class="form-select") }}</div>
                                <div class="col-md-6">{{ form.specialty.label(class="form-label fw-semibold") }} {{ form.specialty(class="form-select") }}</div>
                                <div class="col-12">{{ form.subspecialty.label(class="form-label fw-semibold") }} {{ form.subspecialty(class="form-control") }}</div>
                                <div class="col-md-6">{{ form.first_name.label(class="form-label fw-semibold") }} {{ form.first_name(class="form-control") }}</div>
                                <div class="col-md-6">{{ form.last_name.label(class="form-label fw-semibold") }} {{ form.last_name(class="form-control") }}</div>
                                <div class="col-md-6">{{ form.email.label(class="form-label fw-semibold") }} {{ form.email(class="form-control") }}</div>
                                <div class="col-md-6">{{ form.phone.label(class="form-label fw-semibold") }} {{ form.phone(class="form-control") }}</div>
                                <div class="col-md-6">{{ form.alt_phone.label(class="form-label fw-semibold") }} {{ form.alt_phone(class="form-control") }}</div>
                                <div class="col-12">
                                    <label class="form-label fw-semibold">{{ form.address.label.text }}</label>
                                    {{ form.address(class="form-control", placeholder="Street address (Line 1)") }}
                                </div>
                                <div class="col-md-6">
                                    <label class="form-label fw-semibold">{{ form.city.label.text }}</label>
                                    {{ form.city(class="form-control", placeholder="City") }}
                                </div>
                                <div class="col-md-6">
                                    <label class="form-label fw-semibold">{{ form.state.label.text }}</label>
                                    {{ form.state(class="form-select") }}
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="wizard-step" data-step="1">
                    <div class="row g-4">
                        <div class="col-lg-6 position-section md-do-fields">
                            <p class="section-title">MD/DO Education</p>
                            <div class="mb-3">{{ form.medical_school.label }} {{ form.medical_school(class="form-control") }}</div>
                            <div class="mb-3">{{ form.med_grad_month_year.label }} {{ form.med_grad_month_year(class="form-control") }}</div>
                            <div class="mb-3">{{ form.residency.label }} {{ form.residency(class="form-control") }}</div>
                            <div class="mb-3">{{ form.residency_grad_month_year.label }} {{ form.residency_grad_month_year(class="form-control") }}</div>

                            <p class="section-title">Fellowships</p>
                            <div class="mb-3">
                                {{ form.num_fellowships.label }} {{ form.num_fellowships(class="form-select", id="num_fellowships") }}
                            </div>
                            <div id="fellowship_fields">
                                {% for fellowship_field, date_field in zip(form.fellowship, form.fellowship_grad_month_year) %}
                                <div class="border p-3 mb-3 rounded fellowship-case">
                                    <div class="mb-2">{{ fellowship_field.label }} {{ fellowship_field(class="form-control") }}</div>
                                    <div class="mb-2">{{ date_field.label }} {{ date_field(class="form-control") }}</div>
                                </div>
                                {% endfor %}
                            </div>
                        </div>

                        <div class="col-lg-6 position-section np-pa-fields">
                            <p class="section-title">NP/PA Education</p>
                            <div class="mb-3">{{ form.bachelors.label }} {{ form.bachelors(class="form-control") }}</div>
                            <div class="mb-3">{{ form.bachelors_grad_month_year.label }} {{ form.bachelors_grad_month_year(class="form-control") }}</div>
                            <div class="mb-3">{{ form.msn.label }} {{ form.msn(class="form-control") }}</div>
                            <div class="mb-3">{{ form.msn_grad_month_year.label }} {{ form.msn_grad_month_year(class="form-control") }}</div>
                            <div class="mb-3">{{ form.dnp.label }} {{ form.dnp(class="form-control") }}</div>
                            <div class="mb-3">{{ form.dnp_grad_month_year.label }} {{ form.dnp_grad_month_year(class="form-control") }}</div>
                        </div>
                    </div>
                    <div class="mt-4">
                        <p class="section-title">Malpractice Cases</p>
                        <div class="mb-3">
                            {{ form.num_malpractice_cases.label }} {{ form.num_malpractice_cases(class="form-select", id="num_malpractice_cases") }}
                        </div>
                        <div id="malpractice_fields" class="row g-3">
                            {% for case in form.malpractice_cases %}
                            <div class="col-md-6">
                                <div class="border p-3 rounded malpractice-case h-100">
                                    <div class="mb-2">{{ case.incident_year.label }} {{ case.incident_year(class="form-control") }}</div>
                                    <div class="mb-2">{{ case.outcome.label }} {{ case.outcome(class="form-select") }}</div>
                                    <div class="mb-2">{{ case.payout_amount.label }} {{ case.payout_amount(class="form-control") }}</div>
                                    <div class="mb-2">{{ case.case_explanation.label }} {{ case.case_explanation(class="form-control", rows=3) }}</div>
                                </div>
                            </div>
                            {% endfor %}
                        </div>
                    </div>
                </div>

                <div class="wizard-step" data-step="2">
                    <div class="row g-4">
                        <div class="col-lg-6">
                            <p class="section-title">Board & Clinical Status</p>
                            <div class="mb-3">{{ form.certification.label }} {{ form.certification(class="form-select") }}</div>
                            <div class="mb-3">{{ form.certification_specialty_area.label }} {{ form.certification_specialty_area(class="form-control") }}</div>
                            <div class="mb-3">{{ form.clinically_active.label }} {{ form.clinically_active(class="form-select", id="clinically_active") }}</div>

                            <p class="section-title">Digital & Facility Experience</p>
                            <div class="mb-3">
                                <label class="form-label"><strong>{{ form.emr.label }}</strong></label>
                                <div class="d-flex flex-wrap border rounded p-2" style="max-height:300px; overflow-y:auto;">
                                    {% for emr_option in form.emr %}
                                    <div class="form-check me-3" style="width:200px;">{{ emr_option(class="form-check-input") }} {{ emr_option.label(class="form-check-label") }}</div>
                                    {% endfor %}
                                </div>
                                <div class="mt-2">{{ form.emr_other.label(class="form-label") }} {{ form.emr_other(class="form-control", placeholder="Enter other EMR systems") }}</div>
                            </div>

                            <p class="section-title">Practice Details</p>
                            <div class="mb-3">{{ form.additional_training.label }} {{ form.additional_training(class="form-control") }}</div>
                            <div class="form-check mb-3">{{ form.sponsorship_needed(class="form-check-input") }} {{ form.sponsorship_needed.label(class="form-check-label") }}</div>
                        </div>

                        <div class="col-lg-6">
                            <p class="section-title">Compensation</p>
                            <div class="row g-3 mb-3">
                                <div class="col-md-8">{{ form.salary_expectations.label(class="form-label fw-semibold") }} {{ form.salary_expectations(class="form-control") }}</div>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="wizard-step" data-step="3">
                    <div class="row g-4">
                        <div class="col-lg-6">
                            <p class="section-title">Communication</p>
                            <div class="mb-3">
                                <label class="form-label"><strong>{{ form.languages.label }}</strong></label>
                                <div class="d-flex flex-wrap border rounded p-2" style="max-height:320px; overflow-y:auto;">
                                    {% for language in form.languages %}
                                    <div class="form-check me-3" style="width:180px;">{{ language(class="form-check-input") }} {{ language.label(class="form-check-label") }}</div>
                                    {% endfor %}
                                </div>
                                <div class="mt-2">{{ form.language_other.label(class="form-label") }} {{ form.language_other(class="form-control", placeholder="Enter other languages") }}</div>
                            </div>
                        </div>
                        <div class="col-lg-6">
                            <p class="section-title">Licensure</p>
                            <div class="mb-3">
                                <label class="form-label"><strong>{{ form.states_licensed.label }}</strong></label>
                                <div class="d-flex flex-wrap border rounded p-2" style="max-height:320px; overflow-y:auto;">
                                    {% for state in form.states_licensed %}
                                    <div class="form-check me-3" style="width:100px;">{{ state(class="form-check-input") }} {{ state.label(class="form-check-label") }}</div>
                                    {% endfor %}
                                </div>
                            </div>
                            <div class="mb-3">
                                <label class="form-label"><strong>{{ form.states_willing_to_work.label }}</strong></label>
                                <div class="d-flex flex-wrap border rounded p-2" style="max-height:320px; overflow-y:auto;">
                                    {% for state in form.states_willing_to_work %}
                                    <div class="form-check me-3" style="width:100px;">{{ state(class="form-check-input") }} {{ state.label(class="form-check-label") }}</div>
                                    {% endfor %}
                                </div>
                            </div>
                        </div>
                    </div>

                </div>

                <div class="d-flex justify-content-between align-items-center mt-4">
                    <button type="button" class="btn btn-outline-secondary" id="prevStep" disabled>Back</button>
                    <div class="d-flex gap-2">
                        <button type="button" class="btn btn-primary" id="nextStep">Next</button>
                        {{ form.submit(class="btn btn-success d-none", id="submitWizard") }}
                    </div>
                </div>
            </form>
        </div>

        <script src="https://cdnjs.cloudflare.com/ajax/libs/cropperjs/1.5.13/cropper.min.js"></script>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/cropperjs/1.5.13/cropper.min.css"/>

        <script>
            const input = document.getElementById('profileInput');
            const preview = document.getElementById('preview');
            const cropBtn = document.getElementById('cropBtn');
            const cropContainer = document.getElementById('crop-container');
            let cropper;

            input.addEventListener('change', function (e) {
                const file = e.target.files[0];
                if (file) {
                    const reader = new FileReader();
                    reader.onload = function (e) {
                        preview.src = e.target.result;
                        cropContainer.style.display = 'block';
                        preview.style.display = 'block';
                        if (cropper) cropper.destroy();
                        cropper = new Cropper(preview, {
                            aspectRatio: 1,
                            viewMode: 1,
                            autoCropArea: 1,
                            background: false,
                            guides: false,
                            highlight: false,
                            dragMode: 'move',
                            cropBoxMovable: false,
                            cropBoxResizable: false,
                            toggleDragModeOnDblclick: false,
                            ready() {
                                const cropBoxData = cropper.getCropBoxData();
                                const containerData = cropper.getContainerData();
                                cropper.setCropBoxData({
                                    left: (containerData.width - cropBoxData.width) / 2,
                                    top: (containerData.height - cropBoxData.height) / 2,
                                    width: 260,
                                    height: 260
                                });
                            }
                        });
                        cropBtn.style.display = 'inline-block';
                    }
                    reader.readAsDataURL(file);
                }
            });

            cropBtn.addEventListener('click', function () {
                if (cropper) {
                    const canvas = cropper.getCroppedCanvas({
                        width: 300,
                        height: 300,
                        imageSmoothingEnabled: true,
                        imageSmoothingQuality: 'high'
                    });
                    document.getElementById('croppedImageData').value = canvas.toDataURL();
                    cropBtn.innerText = "Image Cropped";
                }
            });

            function toggleFields(selectorId, classSelector) {
                const sel = document.getElementById(selectorId),
                    fields = document.querySelectorAll(classSelector);
                sel.addEventListener('change', () => {
                    const n = parseInt(sel.value, 10);
                    fields.forEach((el, i) => el.style.display = i < n ? 'block' : 'none');
                });
                sel.dispatchEvent(new Event('change'));
            }

            toggleFields('num_malpractice_cases', '.malpractice-case');
            toggleFields('num_fellowships', '.fellowship-case');

            const positionSelect = document.getElementById('position');
            const mdDoSections = document.querySelectorAll('.md-do-fields');
            const npPaSections = document.querySelectorAll('.np-pa-fields');

            function syncPositionSections() {
                const pos = positionSelect.value;
                const showMdDo = pos === 'MD' || pos === 'DO';
                const showNpPa = pos === 'NP' || pos === 'PA';

                mdDoSections.forEach(el => el.style.display = showMdDo ? '' : 'none');
                npPaSections.forEach(el => el.style.display = showNpPa ? '' : 'none');
            }

            positionSelect.addEventListener('change', syncPositionSections);
            syncPositionSections();

            document.getElementById('clinically_active').addEventListener('change', function () {
                const selectedOption = this.value;
                const lastActiveDiv = document.getElementById('last_active_field');
                if (selectedOption === 'No') {
                    lastActiveDiv.style.display = 'block';
                } else {
                    lastActiveDiv.style.display = 'none';
                    document.getElementById("last_clinically_active").value = '';
                }
            });
            document.getElementById('clinically_active').dispatchEvent(new Event('change'));

            const steps = Array.from(document.querySelectorAll('.wizard-step'));
            const chips = Array.from(document.querySelectorAll('.step-chip'));
            const progressBar = document.getElementById('progressBar');
            const nextBtn = document.getElementById('nextStep');
            const prevBtn = document.getElementById('prevStep');
            const submitBtn = document.getElementById('submitWizard');
            let currentStep = 0;

            function updateStepDisplay() {
                steps.forEach((step, idx) => step.classList.toggle('active', idx === currentStep));
                chips.forEach((chip, idx) => chip.classList.toggle('active', idx === currentStep));
                const progress = ((currentStep + 1) / steps.length) * 100;
                progressBar.style.width = `${progress}%`;
                prevBtn.disabled = currentStep === 0;
                nextBtn.classList.toggle('d-none', currentStep === steps.length - 1);
                submitBtn.classList.toggle('d-none', currentStep !== steps.length - 1);
            }

            nextBtn.addEventListener('click', () => {
                if (currentStep < steps.length - 1) {
                    currentStep += 1;
                    updateStepDisplay();
                }
            });

            prevBtn.addEventListener('click', () => {
                if (currentStep > 0) {
                    currentStep -= 1;
                    updateStepDisplay();
                }
            });

            updateStepDisplay();
        </script>
        {% endblock %}
        ''',


    'view_job.html': '''
    {% extends "base.html" %}
    {% block content %}
    <div class="card shadow p-4">
        <h3>{{ job.title }}</h3>
        <p><strong>Location:</strong> {{ job.location }}</p>
        <p><strong>Salary:</strong> {{ job.salary }}</p>
        <p><strong>Description:</strong><br>{{ job.description }}</p>

        {% if already_interested %}
            <div class="alert alert-info mt-3">You've already expressed interest in this job.</div>
            <button class="btn btn-secondary mt-2" disabled>Interest Already Expressed</button>
        {% else %}
            <form method="post">
                <button class="btn btn-success mt-2">Express Interest</button>
            </form>
        {% endif %}

        <a href="{{ url_for('doctor_jobs') }}" class="btn btn-secondary mt-3">Back to Jobs</a>
    </div>
    {% endblock %}''',

    'inbox.html': '''{% extends 'base.html' %}
            {% block content %}
                <h2>{{ title }}</h2>
                <table class="table table-striped">
                    <thead>
                        <tr>
                            <th>From</th>
                            <th>Message</th>
                            <th>Timestamp</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for msg in messages %}
                        <tr>
                            <td>{{ msg.sender.username }}</td>
                            <td>
                                {{ msg.content }}
                                {% if msg.message_type == 'interest' %}
                                    <a href="{{ url_for('send_invite', doctor_id=msg.doctor_id, job_id=msg.job_id) }}" class="btn btn-primary btn-sm">Send Call Invite</a>
                                    <a href="{{ url_for('doctor_profile', doctor_id=msg.doctor_id) }}" class="btn btn-info btn-sm">View Doctor Profile</a>
                                {% endif %}
                            </td>
                            <td>{{ msg.timestamp.strftime('%Y-%m-%d %H:%M') }}</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            {% endblock %}''',

    'doctor_call_details.html': '''
        {% extends "base.html" %}
        {% block content %}
        <h2>Scheduled Call Details</h2>

        <p><strong>With:</strong> {{ call.scheduled_by.username }} ({{ call.scheduled_by.role }})</p>
        <p><strong>Date & Time:</strong> {{ call.datetime }}</p>
        <p><strong>Reason:</strong> {{ call.reason }}</p>
        <p><strong>Status:</strong> {% if call.canceled %}Canceled{% elif call.reschedule_requested %}Reschedule Requested{% else %}Scheduled{% endif %}</p>

        {% if not call.canceled %}
        <form method="post">
            <button name="action" value="cancel" class="btn btn-danger">Cancel Meeting</button>
        </form>

        <h3 class="mt-4">Request Reschedule</h3>
        <form method="post">
            <div class="mb-3">
                <label>New Date & Time:</label>
                <input type="datetime-local" name="reschedule_datetime" class="form-control" required>
            </div>
            <div class="mb-3">
                <label>Reason for Reschedule:</label>
                <textarea name="reschedule_note" class="form-control"></textarea>
            </div>
            <button name="action" value="reschedule" class="btn btn-warning">Request Reschedule</button>
        </form>
        {% endif %}
        {% endblock %}''',

    'handle_reschedule.html': '''
        {% extends "base.html" %}
        {% block content %}
        <h2>Handle Reschedule Request</h2>

        <p><strong>Doctor:</strong> {{ call.doctor.first_name }} {{ call.doctor.last_name }}</p>
        <p><strong>Current Date & Time:</strong> {{ call.datetime }}</p>
        <p><strong>Requested Date & Time:</strong> {{ call.reschedule_datetime }}</p>
        <p><strong>Reason for Reschedule:</strong> {{ call.reschedule_note }}</p>

        <form method="post">
            <button name="action" value="accept" class="btn btn-success">Accept Request</button>
            <button name="action" value="reject" class="btn btn-danger">Reject Request</button>
        </form>
        {% endblock %}''',

    'create_account.html': '''{% extends "base.html" %}
    {% block content %}
    <h2 class="text-center mb-4">Create Your Account</h2>
    <p class="text-center">Select your role and fill out the form below to join our platform.</p>

    <ul class="nav nav-tabs" id="accountTabs" role="tablist">
        <li class="nav-item" role="presentation">
            <button class="nav-link active" id="doctor-tab" data-bs-toggle="tab" data-bs-target="#doctor" type="button" role="tab">Doctor</button>
        </li>
        <li class="nav-item" role="presentation">
            <button class="nav-link" id="client-tab" data-bs-toggle="tab" data-bs-target="#client" type="button" role="tab">Hospital</button>
        </li>
    </ul>

    <div class="tab-content border p-4" id="accountTabsContent">
        <!-- Doctor Form -->
        <div class="tab-pane fade show active" id="doctor" role="tabpanel">
            <form method="post" action="/public_register_doctor" class="mt-3">
                <div class="mb-3"><input name="first_name" class="form-control" placeholder="First Name"></div>
                <div class="mb-3"><input name="last_name" class="form-control" placeholder="Last Name"></div>
                <div class="mb-3"><input name="specialty" class="form-control" placeholder="Specialty"></div>
                <div class="mb-3"><input name="email" class="form-control" placeholder="Email"></div>
                <div class="mb-3"><input name="username" class="form-control" placeholder="Username"></div>
                <div class="mb-3"><input type="password" name="password" class="form-control" placeholder="Password"></div>
                <div class="mb-3"><input type="password" name="confirm_password" class="form-control" placeholder="Confirm Password"></div>
                <div class="mb-3"><textarea name="reason" class="form-control" placeholder="Reason for joining (optional)"></textarea></div>
                <button type="submit" class="btn btn-success">Create Doctor Account</button>
            </form>
        </div>

        <!-- Client Form -->
        <div class="tab-pane fade" id="client" role="tabpanel">
            <form method="post" action="/public_register_client" class="mt-3">
                <div class="mb-3"><input name="organization_name" class="form-control" placeholder="Hospital or Organization Name"></div>
                <div class="mb-3"><input name="email" class="form-control" placeholder="Email"></div>
                <div class="mb-3"><input name="username" class="form-control" placeholder="Username"></div>
                <div class="mb-3"><input type="password" name="password" class="form-control" placeholder="Password"></div>
                <div class="mb-3"><input type="password" name="confirm_password" class="form-control" placeholder="Confirm Password"></div>
                <div class="mb-3"><textarea name="reason" class="form-control" placeholder="Reason for joining (optional)"></textarea></div>
                <button type="submit" class="btn btn-info">Create Hospital Account</button>
            </form>
        </div>
    </div>

    <div class="text-center mt-4">
        <a href="{{ url_for('login') }}" class="btn btn-secondary">Back to Login</a>
    </div>

    <!-- Bootstrap Tabs -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
    {% endblock %}
    ''',

    'doctors.html': '''{% extends "base.html" %}{% block content %}
        <h2>Doctors in System</h2>
        <table class="table table-striped">
            <tr>
                <th>Name</th>
                <th>Email</th>
                <th>Phone</th>
                <th>Specialty</th>
                <th>States Licensed</th>
                <th>States Willing to Work</th>
                <th>Salary Expectation (Total Compensation)</th>
                <th>Edit</th>
            </tr>
            {% for doctor in doctors %}
            <tr>
                <td><a href="{{ url_for('doctor_profile', doctor_id=doctor.id) }}">{{ doctor.first_name }} {{ doctor.last_name }}</a></td>
                <td>{{ doctor.email }}</td>
                <td>{{ doctor.phone }}</td>
                <td>{{ doctor.specialty }}</td>
                <td>{{ doctor.states_licensed.replace(",", ", ") if doctor.states_licensed else '' }}</td>
                <td>{{ doctor.states_willing_to_work.replace(",", ", ") if doctor.states_willing_to_work else '' }}</td>
                <td>${{ doctor.salary_expectations }}</td>
                <td>
                    <a href="{{ url_for('edit_doctor', doctor_id=doctor.id) }}" class="btn btn-warning btn-sm">Edit</a>
                </td>
            </tr>
            {% endfor %}
        </table>
    {% endblock %}''',

    'edit_call.html': '''{% extends "base.html" %}
    {% block content %}
        <h2>Edit Scheduled Call</h2>
        <form method="post">
            {{ form.hidden_tag() }}

            <div class="mb-3">
                {{ form.doctor_id.label }}
                {{ form.doctor_id(class="form-select") }}
            </div>

            <div class="mb-3">
                {{ form.datetime.label }}
                {{ form.datetime(class="form-control", id="datetime") }}
            </div>

            <div class="mb-3">
                {{ form.reason.label }}
                {{ form.reason(class="form-control") }}
            </div>

            {{ form.submit(class="btn btn-success") }}
        </form>

        <!-- Include Select2 CSS and JS -->
        <link href="https://cdn.jsdelivr.net/npm/select2@4.1.0-rc.0/dist/css/select2.min.css" rel="stylesheet" />
        <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/select2@4.1.0-rc.0/dist/js/select2.min.js"></script>

        <script>
            $(document).ready(function() {
                $('.form-select').select2({
                    placeholder: "Select Doctor (Name | Email | Specialty)",
                    allowClear: true,
                    width: '100%'
                });
            });
            flatpickr("#datetime", {enableTime: true, dateFormat: "Y-m-d\\TH:i"});
        </script>
    {% endblock %}''',

    'send_job.html': '''{% extends 'base.html' %}
    {% block content %}
        <h2>Send Job to {{ doctor.first_name }} {{ doctor.last_name }}</h2>
        <form method="post">
            <div class="mb-3">
                <label>Select Job:</label>
                <select name="job_id" class="form-select">
                    {% for job in jobs %}
                        <option value="{{ job.id }}">{{ job.title }} ({{ job.location }})</option>
                    {% endfor %}
                </select>
            </div>
            <button class="btn btn-success">Send Job</button>
        </form>
        {% endblock %}''',

    'client_analytics.html': '''{% extends "base.html" %}
        {% block content %}
        <h2 class="mb-4">Job Post Analytics</h2>

        <input class="form-control mb-3" type="text" id="jobSearch" placeholder="Search by job title, location, or description...">

        <div id="jobList">
            {% for job in job_data %}
            <div class="card mb-4 shadow-sm job-card">
                <div class="card-body">
                    <h5 class="card-title">{{ job.title }}</h5>
                    <h6 class="card-subtitle mb-2 text-muted">{{ job.location }}</h6>
                    <p class="card-text text-truncate">{{ job.description }}</p>

                    <p><strong>Interest Count:</strong> {{ job.interest_count }}</p>

                    {% if job.interest_count > 0 %}
                    <a href="{{ url_for('download_job_applicants', job_id=job.id) }}" class="btn btn-sm btn-outline-success mb-2">
                        <i class="bi bi-file-earmark-arrow-down"></i> Download Excel
                    </a>
                    <button class="btn btn-sm btn-outline-info mb-2" data-bs-toggle="collapse" data-bs-target="#doctors-{{ job.id }}">View Interested Doctors</button>
                    <div class="collapse" id="doctors-{{ job.id }}">
                        <ul class="list-group">
                            {% for doc in job.interested_doctors %}
                            <li class="list-group-item d-flex justify-content-between align-items-center">
                                <span>
                                    {{ doc.name }} – {{ doc.email }}
                                </span>
                                <span>
                                    {% if doc.id %}
                                        <a href="{{ url_for('doctor_profile', doctor_id=doc.id, job_id=job.id) }}"
                                        class="btn btn-sm btn-outline-primary me-2">View Profile</a>
                                        <a href="{{ url_for('send_invite', doctor_id=doc.id, job_id=job.id) }}"
                                        class="btn btn-sm btn-success">Schedule Call</a>
                                    {% else %}
                                        <span class="text-muted">No profile available</span>
                                    {% endif %}
                                </span>
                            </li>
                            {% endfor %}
                        </ul>
                    </div>

                    <canvas id="chart-{{ job.id }}" height="100"></canvas>
                    <script>
                        const ctx{{ job.id }} = document.getElementById('chart-{{ job.id }}').getContext('2d');
                        new Chart(ctx{{ job.id }}, {
                            type: 'bar',
                            data: {
                                labels: {{ job.interest_by_day.keys() | list | tojson }},
                                datasets: [{
                                    label: 'Interest Count by Day',
                                    data: {{ job.interest_by_day.values() | list | tojson }},
                                    backgroundColor: 'rgba(0, 123, 255, 0.5)',
                                    borderColor: 'rgba(0, 123, 255, 1)',
                                    borderWidth: 1
                                }]
                            },
                            options: {
                                scales: {
                                    x: { title: { display: true, text: 'Date' } },
                                    y: { beginAtZero: true, title: { display: true, text: 'Interests' } }
                                }
                            }
                        });
                    </script>
                    {% else %}
                    <p class="text-muted">No interest yet.</p>
                    {% endif %}
                </div>
            </div>
            {% endfor %}
        </div>

        <a href="{{ url_for('client_dashboard') }}" class="btn btn-secondary mt-4">Back to Dashboard</a>

        <!-- Chart.js + live search -->
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script>
        document.getElementById('jobSearch').addEventListener('keyup', function () {
            const query = this.value.toLowerCase();
            document.querySelectorAll('.job-card').forEach(card => {
                const text = card.innerText.toLowerCase();
                card.style.display = text.includes(query) ? '' : 'none';
            });
        });
        </script>
        {% endblock %}''',

    'edit_doctor.html': '''{% extends \"base.html\" %}
        {% block content %}
        <h2>Edit Doctor: {{ doctor.first_name }} {{ doctor.last_name }}</h2>
        <form method=\"post\">
            {{ form.hidden_tag() }}

            <div class=\"mb-3\">{{ form.position.label }} {{ form.position(class=\"form-select\") }}</div>
            <div class=\"mb-3\">{{ form.specialty.label }} {{ form.specialty(class=\"form-select\") }}</div>
            <div class=\"mb-3\">{{ form.subspecialty.label }} {{ form.subspecialty(class=\"form-control\") }}</div>
            <div class=\"mb-3\">{{ form.first_name.label }} {{ form.first_name(class=\"form-control\") }}</div>
            <div class=\"mb-3\">{{ form.last_name.label }} {{ form.last_name(class=\"form-control\") }}</div>
            <div class=\"mb-3\">{{ form.email.label }} {{ form.email(class=\"form-control\") }}</div>
            <div class=\"mb-3\">{{ form.phone.label }} {{ form.phone(class=\"form-control\") }}</div>
            <div class=\"mb-3\">{{ form.alt_phone.label }} {{ form.alt_phone(class=\"form-control\") }}</div>
            <div class=\"mb-3\">\n                <label class=\"form-label fw-semibold\">{{ form.address.label.text }}</label>\n                {{ form.address(class=\"form-control\", placeholder=\"Street address (Line 1)\") }}\n            </div>
            <div class=\"row mb-3 g-3\">\n                <div class=\"col-md-6\">\n                    <label class=\"form-label fw-semibold\">{{ form.city.label.text }}</label>\n                    {{ form.city(class=\"form-control\", placeholder=\"City\") }}\n                </div>\n                <div class=\"col-md-6\">\n                    <label class=\"form-label fw-semibold\">{{ form.state.label.text }}</label>\n                    {{ form.state(class=\"form-select\") }}\n                </div>\n            </div>

            <h4>MD/DO Information</h4>
            <div class=\"mb-3\">{{ form.medical_school.label }} {{ form.medical_school(class=\"form-control\") }}</div>
            <div class=\"mb-3\">{{ form.med_grad_month_year.label }} {{ form.med_grad_month_year(class=\"form-control\") }}</div>
            <div class=\"mb-3\">{{ form.residency.label }} {{ form.residency(class=\"form-control\") }}</div>
            <div class=\"mb-3\">{{ form.residency_grad_month_year.label }} {{ form.residency_grad_month_year(class=\"form-control\") }}</div>

            <h4>Fellowships</h4>
            <div class=\"mb-3\">
                {{ form.num_fellowships.label }}
                {{ form.num_fellowships(class=\"form-select\", id=\"num_fellowships\") }}
            </div>

            <div id=\"fellowship_fields\">
                {% for fellowship_field, date_field in zip(form.fellowship, form.fellowship_grad_month_year) %}
                <div class=\"border p-3 mb-3 rounded fellowship-case\">
                    <div class=\"mb-2\">
                        {{ fellowship_field.label }} {{ fellowship_field(class=\"form-control\") }}
                    </div>
                    <div class=\"mb-2\">
                        {{ date_field.label }} {{ date_field(class=\"form-control\") }}
                    </div>
                </div>
                {% endfor %}
            </div>

            <h4>NP/PA Information</h4>
            <div class=\"mb-3\">{{ form.bachelors.label }} {{ form.bachelors(class=\"form-control\") }}</div>
            <div class=\"mb-3\">{{ form.bachelors_grad_month_year.label }} {{ form.bachelors_grad_month_year(class=\"form-control\") }}</div>
            <div class=\"mb-3\">{{ form.msn.label }} {{ form.msn(class=\"form-control\") }}</div>
            <div class=\"mb-3\">{{ form.msn_grad_month_year.label }} {{ form.msn_grad_month_year(class=\"form-control\") }}</div>
            <div class=\"mb-3\">{{ form.dnp.label }} {{ form.dnp(class=\"form-control\") }}</div>
            <div class=\"mb-3\">{{ form.dnp_grad_month_year.label }} {{ form.dnp_grad_month_year(class=\"form-control\") }}</div>
            <div class=\"mb-3\">{{ form.additional_training.label }} {{ form.additional_training(class=\"form-control\") }}</div>
            <div class=\"form-check mb-3\">{{ form.sponsorship_needed(class=\"form-check-input\") }} {{ form.sponsorship_needed.label(class=\"form-check-label\") }}</div>

            <h4>Malpractice Cases</h4>
            <div class=\"mb-3\">
                {{ form.num_malpractice_cases.label }}
                {{ form.num_malpractice_cases(class=\"form-select\", id=\"num_malpractice_cases\") }}
            </div>
            <div id=\"malpractice_fields\">
                {% for case in form.malpractice_cases %}
                <div class=\"border p-3 mb-3 rounded malpractice-case\">
                    <div class=\"mb-2\">{{ case.incident_year.label }} {{ case.incident_year(class=\"form-control\") }}</div>
                    <div class=\"mb-2\">{{ case.outcome.label }} {{ case.outcome(class=\"form-select\") }}</div>
                    <div class=\"mb-2\">{{ case.payout_amount.label }} {{ case.payout_amount(class=\"form-control\") }}</div>
                    <div class=\"mb-2\">{{ case.case_explanation.label }} {{ case.case_explanation(class=\"form-control\", rows=3) }}</div>
                </div>
                {% endfor %}
            </div>


            <div class=\"mb-3\">{{ form.certification.label }} {{ form.certification(class=\"form-select\") }}</div>
            <div class="mb-3">{{ form.certification_specialty_area.label }} {{ form.certification_specialty_area(class="form-control") }}</div>
            <div class="mb-3">
                {{ form.clinically_active.label }} {{ form.clinically_active(class="form-select", id="clinically_active") }}
            </div>

            <div class="mb-3" id="last_active_field" style="display:none;">
                {{ form.last_clinically_active.label }} {{ form.last_clinically_active(class="form-control") }}
            </div>
            <div class=\"row mb-3\">\n                <div class=\"col-md-6 mb-3 mb-md-0\">\n                    <label class=\"form-label\"><strong>{{ form.emr.label }}</strong></label>\n                    <div class=\"d-flex flex-wrap border rounded p-2\" style=\"max-height:300px; overflow-y:auto;\">\n                        {% for emr_option in form.emr %}\n                        <div class=\"form-check me-3\" style=\"width:200px;\">{{ emr_option(class=\"form-check-input\") }} {{ emr_option.label(class=\"form-check-label\") }}</div>\n                        {% endfor %}\n                    </div>\n                </div>\n                <div class=\"col-md-6\">\n                    <label class=\"form-label\"><strong>{{ form.languages.label }}</strong></label>\n                    <div class=\"d-flex flex-wrap border rounded p-2\" style=\"max-height:300px; overflow-y:auto;\">\n                        {% for language in form.languages %}\n                        <div class=\"form-check me-3\" style=\"width:180px;\">{{ language(class=\"form-check-input\") }} {{ language.label(class=\"form-check-label\") }}</div>\n                        {% endfor %}\n                    </div>\n                </div>\n            </div>

           <div class="row mb-3">
                <div class="col-md-6">
                    <label class="form-label"><strong>{{ form.states_licensed.label }}</strong></label>
                    <div class="d-flex flex-wrap border rounded p-2" style="max-height:300px; overflow-y:auto;">
                        {% for state in form.states_licensed %}
                        <div class="form-check me-3" style="width:100px;">
                            {{ state(class="form-check-input") }} 
                            {{ state.label(class="form-check-label") }}
                        </div>
                        {% endfor %}
                    </div>
                </div>
                <div class="col-md-6">
                    <label class="form-label"><strong>{{ form.states_willing_to_work.label }}</strong></label>
                    <div class="d-flex flex-wrap border rounded p-2" style="max-height:300px; overflow-y:auto;">
                        {% for state in form.states_willing_to_work %}
                        <div class="form-check me-3" style="width:100px;">
                            {{ state(class="form-check-input") }} 
                            {{ state.label(class="form-check-label") }}
                        </div>
                        {% endfor %}
                    </div>
                </div>
            </div>

            <div class=\"mb-3\">{{ form.salary_expectations.label }} {{ form.salary_expectations(class=\"form-control\") }}</div>

            {{ form.submit(class=\"btn btn-success\") }}
        </form>

        
        <script>
            function toggleFields(selectorId, classSelector) {
                const sel = document.getElementById(selectorId),
                    fields = document.querySelectorAll(classSelector);
                sel.addEventListener('change', () => {
                    const n = parseInt(sel.value, 10);
                    fields.forEach((el, i) => el.style.display = i < n ? 'block' : 'none');
                });
                sel.dispatchEvent(new Event('change'));
            }

            toggleFields('num_malpractice_cases', '.malpractice-case');
            toggleFields('num_fellowships', '.fellowship-case');

            // Your new clinically active script below
            document.getElementById('clinically_active').addEventListener('change', function() {
                const selectedOption = this.value;
                const lastActiveDiv = document.getElementById('last_active_field');

                if (selectedOption === 'No') {
                    lastActiveDiv.style.display = 'block';
                } else {
                    lastActiveDiv.style.display = 'none';
                    document.getElementById("last_clinically_active").value = ''; // Clear input if hidden
                }
            });

            // Trigger the event on page load (to maintain state on edit pages)
            document.getElementById('clinically_active').dispatchEvent(new Event('change'));
        </script>
        {% endblock %}'''
         })

# Routes

@app.route('/send_job/<int:doctor_id>', methods=['GET', 'POST'])
@login_required
def send_job_to_doctor(doctor_id):
    if current_user.role not in ['admin', 'client']:
        flash('Unauthorized', 'danger')
        return redirect(url_for('home'))

    doctor = Doctor.query.get_or_404(doctor_id)
    doctor_user = User.query.get(doctor.user_id)

    # Ensure clients can only see their own jobs
    if current_user.role == 'admin':
        jobs = Job.query.all()
    else:  # client
        jobs = Job.query.filter_by(poster_id=current_user.id).all()

    if request.method == 'POST':
        job_id = request.form.get('job_id')
        job = Job.query.get_or_404(job_id)

        # Confirm clients only send their own job posts
        if current_user.role == 'client' and job.poster_id != current_user.id:
            flash('You can only send your own jobs.', 'danger')
            return redirect(url_for('home'))

        message = Message(
            sender_id=current_user.id,
            recipient_id=doctor_user.id,
            job_id=job.id,
            content=f"{current_user.username} has recommended the job: '{job.title}' for you."
        )

        db.session.add(message)
        db.session.commit()
        flash('Job sent to doctor!', 'success')
        return redirect(url_for('doctor_profile', doctor_id=doctor_id))

    return render_template('send_job.html', doctor=doctor, jobs=jobs)



@app.route('/doctor/inbox')
@login_required
def doctor_inbox():
    if current_user.role != 'doctor':
        flash('Unauthorized', 'danger')
        return redirect(url_for('home'))

    messages = Message.query.filter_by(recipient_id=current_user.id).order_by(Message.timestamp.desc()).all()
    
    # Mark unread messages as read
    unread_messages = [msg for msg in messages if not msg.read]
    for msg in unread_messages:
        msg.read = True
    db.session.commit()

    return render_template('inbox.html', messages=messages, title="My Inbox")

@app.route('/admin/inbox')
@login_required
def admin_inbox():
    if current_user.role != 'admin':
        flash('Unauthorized', 'danger')
        return redirect(url_for('home'))

    messages = Message.query.filter_by(recipient_id=current_user.id).order_by(Message.timestamp.desc()).all()

    unread_messages = [msg for msg in messages if not msg.read]
    for msg in unread_messages:
        msg.read = True
    db.session.commit()

    return render_template('inbox.html', messages=messages, title="Admin Inbox")

@app.route('/add_doctor', methods=['GET', 'POST'])
@login_required
def add_doctor():
    if current_user.role != 'admin':
        flash('Unauthorized access!', 'danger')
        return redirect(url_for('home'))

    form = DoctorForm()
    if form.validate_on_submit():
        if User.query.filter_by(email=form.email.data).first() or Doctor.query.filter_by(email=form.email.data).first():
            flash('A user or doctor with this email already exists.', 'danger')
            return redirect(url_for('add_doctor'))

        username_base = form.first_name.data.lower() + form.last_name.data.lower()
        username = username_base
        suffix = 1
        while User.query.filter_by(username=username).first():
            username = f"{username_base}{suffix}"
            suffix += 1

        new_user = User(username=username, email=form.email.data, role='doctor')
        new_user.set_password('TempPass123!')
        db.session.add(new_user)
        db.session.flush()

        profile_path = None
        if form.profile_picture.data:
            picture = form.profile_picture.data
            filename = secure_filename(picture.filename)
            profile_path = f"upload/{filename}"
            picture.save(os.path.join(app.static_folder, profile_path))

        cropped_data = request.form.get('cropped_image_data')
        if cropped_data:
            import base64
            from PIL import Image
            from io import BytesIO

            header, encoded = cropped_data.split(",", 1)
            binary_data = base64.b64decode(encoded)
            img = Image.open(BytesIO(binary_data))

            cropped_filename = f"cropped_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
            cropped_path = os.path.join(app.static_folder, 'upload', cropped_filename)
            img.save(cropped_path)

            profile_path = f"upload/{cropped_filename}"

        emr_selections = list(form.emr.data or [])
        if form.emr_other.data:
            emr_selections.extend([item.strip() for item in form.emr_other.data.split(',') if item.strip()])

        language_selections = list(form.languages.data or [])
        if form.language_other.data:
            language_selections.extend([item.strip() for item in form.language_other.data.split(',') if item.strip()])

        new_doctor = Doctor(
            user_id=new_user.id,
            profile_picture=profile_path,
            position=form.position.data,
            specialty=form.specialty.data,
            subspecialty=form.subspecialty.data,
            first_name=form.first_name.data,
            last_name=form.last_name.data,
            email=form.email.data,
            phone=form.phone.data,
            alt_phone=form.alt_phone.data,
            address=form.address.data,
            city_of_residence=format_city_state(form.city.data, form.state.data),
            medical_school=form.medical_school.data,
            med_grad_month_year=form.med_grad_month_year.data,
            residency=form.residency.data,
            residency_grad_month_year=form.residency_grad_month_year.data,
            fellowship=",".join(form.fellowship.data[:int(form.num_fellowships.data)]),
            fellowship_grad_month_year=",".join(form.fellowship_grad_month_year.data[:int(form.num_fellowships.data)]),
            bachelors=form.bachelors.data,
            bachelors_grad_month_year=form.bachelors_grad_month_year.data,
            msn=form.msn.data,
            msn_grad_month_year=form.msn_grad_month_year.data,
            dnp=form.dnp.data,
            dnp_grad_month_year=form.dnp_grad_month_year.data,
            additional_training=form.additional_training.data,
            sponsorship_needed=form.sponsorship_needed.data or False,
            malpractice_cases=json.dumps([
                {
                    'incident_year': case.form.incident_year.data,
                    'outcome': case.form.outcome.data,
                    'payout_amount': case.form.payout_amount.data or 0,
                    'case_explanation': case.form.case_explanation.data or ''
                } for case in form.malpractice_cases.entries[:int(form.num_malpractice_cases.data)]
            ]),
            certification=form.certification.data,
            certification_specialty_area=form.certification_specialty_area.data,
            clinically_active=form.clinically_active.data,
            last_clinically_active=form.last_clinically_active.data if form.clinically_active.data == 'No' else None,
            emr=",".join(emr_selections),
            languages=",".join(language_selections),
            states_licensed=",".join(form.states_licensed.data),
            states_willing_to_work=",".join(form.states_willing_to_work.data),
            salary_expectations=form.salary_expectations.data or 0.0,
            joined=datetime.utcnow()
        )


        db.session.add(new_doctor)
        db.session.commit()

        flash(f"Doctor profile and user '{username}' created successfully!", 'success')
        return redirect(url_for('doctors'))

    return render_template('add_doctor.html', form=form, zip=zip)




@app.route('/schedule_call', methods=['GET', 'POST'])
@login_required
def schedule_call():
    form = ScheduledCallForm()

    form.doctor_id.choices = [
        (doc.id, f"{doc.first_name} {doc.last_name} | {doc.email} | {doc.specialty}") 
        for doc in Doctor.query.order_by(Doctor.last_name).all()
    ]

    if form.validate_on_submit():
        dt = form.datetime.data  # <-- FIX HERE

        invite_status = "Pending" if request.form.get('send_invite') == 'yes' else "Accepted"

        call = ScheduledCall(
            doctor_id=form.doctor_id.data,
            scheduled_by_id=current_user.id,
            job_id=None,
            datetime=dt,
            reason=form.reason.data,
            invite_status=invite_status
        )
        db.session.add(call)
        db.session.commit()

        if invite_status == "Pending":
            doctor = Doctor.query.get(form.doctor_id.data)
            doctor_user = User.query.get(doctor.user_id)
            if doctor_user:
                message = Message(
                    sender_id=current_user.id,
                    recipient_id=doctor_user.id,
                    content=(
                        f"You have a new call invite scheduled on {dt.strftime('%Y-%m-%d %H:%M')} "
                        f"for reason: {form.reason.data}. Please accept or decline on your dashboard."
                    )
                )
                db.session.add(message)
                db.session.commit()
                flash('Invite sent to doctor successfully!', 'success')
        else:
            flash('Call scheduled successfully!', 'success')

        return redirect(url_for('home'))

    return render_template('schedule_call.html', form=form)



@app.route('/create_account')
def create_account():
    return render_template('create_account.html')


@app.route('/admin/dashboard')
@login_required
def admin_dashboard():
    if current_user.role != 'admin':
        flash('Unauthorized access!', 'danger')
        return redirect(url_for('home'))

    scheduled_calls = ScheduledCall.query.order_by(ScheduledCall.datetime).all()

    events = [
        {
            'id': call.id,
            'title': call.doctor.first_name + " " + call.doctor.last_name + " - " + (call.reason or "No Reason"),
            'start': call.datetime.strftime('%Y-%m-%dT%H:%M:%S'),
        } for call in scheduled_calls
    ]

    return render_template('index.html', events=events)


@app.route('/doctors')
def doctors():
    all_doctors = Doctor.query.all()
    return render_template('doctors.html', doctors=all_doctors)

@app.route('/edit_call/<int:call_id>', methods=['GET', 'POST'])
@login_required
def edit_call(call_id):
    scheduled_call = ScheduledCall.query.get_or_404(call_id)
    form = ScheduledCallForm(obj=scheduled_call)

    form.doctor_id.choices = [
        (doc.id, f"{doc.first_name} {doc.last_name} | {doc.email} | {doc.specialty}") 
        for doc in Doctor.query.order_by(Doctor.last_name).all()
    ]

    if form.validate_on_submit():
        # Check if datetime or doctor has changed
        original_datetime = scheduled_call.datetime
        original_doctor_id = scheduled_call.doctor_id

        scheduled_call.doctor_id = form.doctor_id.data
        scheduled_call.datetime = form.datetime.data if isinstance(form.datetime.data, datetime) else datetime.strptime(form.datetime.data, '%Y-%m-%dT%H:%M')
        scheduled_call.reason = form.reason.data
        
        db.session.commit()

        # Send notification to doctor's inbox if rescheduled
        if (scheduled_call.datetime != original_datetime or 
            scheduled_call.doctor_id != original_doctor_id):

            doctor_user = User.query.filter_by(doctor_id=scheduled_call.doctor_id).first()
            if doctor_user:
                message = Message(
                    sender_id=current_user.id,
                    recipient_id=doctor_user.id,
                    content=(
                        f"The scheduled meeting has been rescheduled to "
                        f"{scheduled_call.datetime.strftime('%Y-%m-%d %H:%M')} "
                        f"by {current_user.username}."
                    )
                )
                db.session.add(message)
                db.session.commit()

        flash('Scheduled call updated successfully!', 'success')
        return redirect(url_for('calls'))

    form.doctor_id.data = scheduled_call.doctor_id
    form.datetime.data = scheduled_call.datetime 
    form.reason.data = scheduled_call.reason

    return render_template('edit_call.html', form=form, call=scheduled_call)





#@app.route('/static/<path:filename>')
#def custom_static(filename):
    #full_path = r'C:\Users\Tubam\OneDrive\Desktop\static'
    #print(f"Serving file: {filename} from path: {full_path}")
    #return send_from_directory(full_path, filename)
@app.route('/')
def index():
    if current_user.is_authenticated:
        if current_user.role == 'doctor':
            return redirect(url_for('doctor_dashboard'))
        elif current_user.role == 'admin':
            return redirect(url_for('dashboard'))
        elif current_user.role == 'client':
            return redirect(url_for('client_dashboard'))
    return render_template('landing_page.html', current_year=datetime.now().year)

@app.route('/calls')
def calls():
    scheduled_calls = ScheduledCall.query.order_by(ScheduledCall.datetime).all()
    events = [
    {
        'id': call.id,
        'title': call.doctor.first_name + " " + call.doctor.last_name + " - " + (call.reason or "No Reason"),
        'start': call.datetime.strftime('%Y-%m-%dT%H:%M:%S'),
    } for call in scheduled_calls
    ]
    return render_template('calls.html', events=events)

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and user.check_password(form.password.data):
            login_user(user)
            flash('Logged in successfully!', 'success')
            
            # Redirect users immediately based on their role
            if user.role == 'doctor':
                return redirect(url_for('doctor_dashboard'))
            elif user.role == 'client':
                return redirect(url_for('client_dashboard'))
            elif user.role == 'admin':
                return redirect(url_for('dashboard'))
            else:
                flash('Role not recognized.', 'danger')
                return redirect(url_for('login'))

        flash('Invalid username or password.', 'danger')

    return render_template('login.html', form=form)


@app.route('/register_doctor', methods=['GET', 'POST'])
@login_required
def register_doctor():
    if current_user.role != 'admin':
        flash('Unauthorized access!', 'danger')
        return redirect(url_for('home'))

    form = DoctorRegistrationForm()
    if form.validate_on_submit():
        existing_user = User.query.filter(
            (User.username == form.username.data) | (User.email == form.email.data)
        ).first()

        if existing_user:
            flash('Username or email already exists for user login.', 'danger')
            return redirect(url_for('register_doctor'))

        # Check if doctor profile already exists with the email provided
        existing_doctor = Doctor.query.filter_by(email=form.email.data).first()

        if existing_doctor:
            # Create user linked to existing doctor profile
            new_user = User(
                username=form.username.data,
                email=form.email.data,
                password_hash=generate_password_hash(form.password.data),
                role='doctor'
            )
            db.session.add(new_user)
            db.session.commit()

            # Associate the existing doctor with the new user's ID
            existing_doctor.user_id = new_user.id
            db.session.commit()

            flash('Doctor login linked to existing profile!', 'success')
            return redirect(url_for('home'))

        # First create new user
        new_user = User(
            username=form.username.data,
            email=form.email.data,
            password_hash=generate_password_hash(form.password.data),
            role='doctor'
        )
        db.session.add(new_user)
        db.session.commit()

        # Then create new doctor profile associated to the new user
        new_doctor = Doctor(
            first_name='',
            last_name='',
            email=form.email.data,
            position='',
            specialty='',
            joined=datetime.utcnow(),
            user_id=new_user.id  # <-- Associate new doctor with new user
        )
        db.session.add(new_doctor)
        db.session.commit()

        # Finally, link newly created doctor profile back to user
        new_user.doctor_id = new_doctor.id
        db.session.commit()

        flash('New Doctor account created and linked!', 'success')
        return redirect(url_for('home'))

    return render_template('register_doctor.html', form=form)


@app.route('/scrape_jobs')
@login_required
def scrape_jobs():
    if current_user.role not in ['user', 'admin']:
        flash('Unauthorized access!', 'danger')
        return redirect(url_for('home'))

    csv_path = Path(__file__).resolve().parent / "DocCafe job export.csv"

    if not csv_path.exists():
        flash(f"DocCafe export not found at {csv_path.name}.", "danger")
        return redirect(url_for('doctor_jobs'))

    required_columns = {
        'Occupation Name', 'Specialty Name', 'State', 'City', 'Job Title',
        'Job Number', 'Job Status', 'Job Link', 'Site'
    }

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        flash(f"Unable to read DocCafe export: {exc}", "danger")
        return redirect(url_for('doctor_jobs'))

    missing = required_columns - set(df.columns)
    if missing:
        flash(f"DocCafe export is missing columns: {', '.join(sorted(missing))}", "danger")
        return redirect(url_for('doctor_jobs'))

    jobs_added = 0

    for _, row in df.iterrows():
        title = str(row.get('Job Title', '')).strip() or 'Untitled Job'

        city_value = row.get('City')
        city = ""
        if pd.notna(city_value):
            city = str(city_value).strip()
            if city.lower() == "nan":
                city = ""

        state = str(row.get('State', '') or '').strip()
        location = format_city_state(city, state)

        description_parts = []
        occupation = row.get('Occupation Name')
        specialty = row.get('Specialty Name')
        job_number = row.get('Job Number')
        job_status = row.get('Job Status')
        job_link = row.get('Job Link')
        site = row.get('Site')
        if occupation:
            description_parts.append(f"Occupation: {occupation}")
        if specialty:
            description_parts.append(f"Specialty: {specialty}")
        if job_number:
            description_parts.append(f"Job Number: {job_number}")
        if job_status:
            description_parts.append(f"Status: {job_status}")
        if job_link:
            description_parts.append(f"Link: {job_link}")
        if site:
            description_parts.append(f"Site: {site}")

        description = " | ".join(description_parts) if description_parts else 'N/A'
        existing_job = Job.query.filter_by(
            title=title,
            location=location,
            poster_id=current_user.id
        ).first()

        if existing_job:
            continue

        lat, lng = geocode_location(location) if location else (None, None)
        new_job = Job(
            title=title,
            location=location,
            salary='',
            description=description,
            poster_id=current_user.id,
            latitude=lat,
            longitude=lng
        )
        db.session.add(new_job)
        db.session.commit()
        jobs_added += 1

    flash(f"{jobs_added} DocCafe job postings added!", "success")
    return redirect(url_for('doctor_jobs'))
@app.route('/post_job', methods=['GET', 'POST'])
@login_required
def post_job():
    if current_user.role not in ['client', 'admin']:
        flash('Only clients and admins can post jobs!', 'danger')
        return redirect(url_for('home'))

    form = JobForm()
    if form.validate_on_submit():
        # Geocode location before creating the Job
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
        flash('Job posted successfully! Add job-specific needs to qualify doctors.', 'success')

        return redirect(url_for('job_requirements', job_id=job.id))

    return render_template('post_job.html', form=form)


@app.route('/job/<int:job_id>/requirements', methods=['GET', 'POST'])
@login_required
def job_requirements(job_id):
    job = Job.query.get_or_404(job_id)

    if current_user.role not in ['client', 'admin']:
        flash('Only clients and admins can manage job requirements!', 'danger')
        return redirect(url_for('home'))

    if current_user.role == 'client' and job.poster_id != current_user.id:
        flash('You can only update requirements for your own jobs.', 'danger')
        return redirect(url_for('home'))

    form = JobRequirementForm()
    requirement = JobRequirement.query.filter_by(job_id=job.id).first()

    if request.method == 'GET' and requirement:
        form.position.data = requirement.position or form.position.data
        form.specialty.data = requirement.specialty or form.specialty.data
        form.subspecialty.data = requirement.subspecialty or ''
        form.certification.data = requirement.certification or ''
        form.certification_specialty_area.data = requirement.certification_specialty_area or ''
        form.clinically_active.data = requirement.clinically_active or ''
        form.emr.data = requirement.emr.split(',') if requirement.emr else []
        form.emr_other.data = requirement.emr_other or ''
        form.languages.data = requirement.languages.split(',') if requirement.languages else []
        form.language_other.data = requirement.language_other or ''
        form.states_required.data = requirement.states_required.split(',') if requirement.states_required else []
        form.states_preferred.data = requirement.states_preferred.split(',') if requirement.states_preferred else []
        form.sponsorship_supported.data = requirement.sponsorship_supported
        form.salary_range.data = requirement.salary_range or ''
        form.notes.data = requirement.notes or ''

    if form.validate_on_submit():
        if not requirement:
            requirement = JobRequirement(job=job)

        requirement.position = form.position.data
        requirement.specialty = form.specialty.data
        requirement.subspecialty = form.subspecialty.data
        requirement.certification = form.certification.data
        requirement.certification_specialty_area = form.certification_specialty_area.data
        requirement.clinically_active = form.clinically_active.data
        requirement.emr = ",".join(form.emr.data) if form.emr.data else None
        requirement.emr_other = form.emr_other.data
        requirement.languages = ",".join(form.languages.data) if form.languages.data else None
        requirement.language_other = form.language_other.data
        requirement.states_required = ",".join(form.states_required.data) if form.states_required.data else None
        requirement.states_preferred = ",".join(form.states_preferred.data) if form.states_preferred.data else None
        requirement.sponsorship_supported = form.sponsorship_supported.data or False
        requirement.salary_range = form.salary_range.data
        requirement.notes = form.notes.data

        db.session.add(requirement)
        db.session.commit()
        flash('Job-specific needs saved! Use them to compare with doctor profiles for qualification.', 'success')

        return redirect(url_for('client_dashboard') if current_user.role == 'client' else url_for('home'))

    return render_template('job_requirements.html', form=form, job=job)

@app.route('/doctor/ai_search_jobs', methods=['POST'])
@login_required
def doctor_ai_search_jobs():
    if current_user.role != 'doctor':
        return jsonify({'error': 'Unauthorized'}), 401

    sync_direct_jobs_from_excel()

    doctor = current_user.doctor
    lifestyle = request.form.get('lifestyle', '')
    wants = request.form.get('wants', '')
    location = (request.form.get('location', '') or '').strip()

    doctor_profile = {
        "name": f"Dr. {doctor.first_name or ''} {doctor.last_name or ''}".strip(),
        "specialty": doctor.specialty or "",
        "subspecialty": doctor.subspecialty or "",
        "home_base": doctor.city_of_residence or "",
        "licensed_states": normalize_state_values((doctor.states_licensed or "").split(',')),
        "preferred_states": normalize_state_values((doctor.states_willing_to_work or "").split(',')),
    }

    # Use stored preferences when the form is blank so the AI has the right context.
    if not location:
        if doctor_profile["preferred_states"]:
            location = ", ".join(doctor_profile["preferred_states"])
        elif doctor_profile["licensed_states"]:
            location = ", ".join(doctor_profile["licensed_states"])
        else:
            location = doctor_profile["home_base"]

    jobs = get_doctor_jobs(doctor)

    jobs_payload = [
        {
            "id": job.id,
            "title": job.title,
            "location": job.location,
            "salary": job.salary,
            "description": job.description
        } for job in jobs
    ]


    pref_terms = [t for t in re.split(r"[^a-z0-9]+", (wants + " " + lifestyle).lower()) if t]
    lifestyle_terms = [t for t in re.split(r"[^a-z0-9]+", lifestyle.lower()) if t]
    wants_terms = [t for t in re.split(r"[^a-z0-9]+", wants.lower()) if t]

    def term_score(text: str, terms):
        if not terms:
            return 0
        text = text.lower()
        return sum(1 for term in terms if term and term in text)


    licensed_states = set(doctor_profile["licensed_states"])
    preferred_states = set(doctor_profile["preferred_states"])
    preferred_location_state = (
        extract_state_abbr(location)
        or STATE_NAME_TO_ABBR.get(re.sub(r"\s+", " ", location).strip().upper(), "")
    ) if location else ""

    scored_jobs = []
    for job in jobs:
        job_text = f"{job.title or ''} {job.description or ''}"
        job_state = extract_state_abbr(job.location)
        state_score = 4 if job_state in licensed_states else 2 if job_state in preferred_states else 0
        location_score = 0
        if preferred_location_state and job_state == preferred_location_state:
            location_score = 5
        elif location and location.lower() in (job.location or '').lower():
            location_score = 4
        lifestyle_score = term_score(job_text, lifestyle_terms)
        wants_score = term_score(job_text, wants_terms)
        overall_score = (
            term_score(job_text, pref_terms) * 2
            + lifestyle_score * 2
            + wants_score * 3
            + location_score
            + state_score
        )
        scored_jobs.append({
            'job': job,
            'overall': overall_score,
            'location_score': location_score,
            'lifestyle_score': lifestyle_score,
            'wants_score': wants_score,
            'state_score': state_score,
        })

    def pick_best(key):
        return max(scored_jobs, key=key) if scored_jobs else None

    best_overall = pick_best(lambda item: item['overall'])
    best_location = pick_best(lambda item: (item['location_score'], item['overall']))
    best_lifestyle = pick_best(lambda item: (item['lifestyle_score'], item['overall']))
    best_wants = pick_best(lambda item: (item['wants_score'], item['overall']))

    def build_reason(item, focus_label: str):
        if not item:
            return "No jobs available to match right now."
        reasons = []
        if item['location_score'] and location:
            reasons.append(f"Matches your location preference for {location}.")
        if item['state_score']:
            reasons.append("Aligns with your licensed or preferred states.")
        if item['wants_score']:
            reasons.append("Includes the job-specific wants you listed.")
        if item['lifestyle_score']:
            reasons.append("Aligns with your lifestyle preferences.")
        if not reasons:
            reasons.append(f"Closest {focus_label.lower()} option available from the current postings.")
        return " ".join(reasons)

    def render_card(title: str, item):
        if not item:
            return f"<h2 style='color:#0b3a82;'>{html.escape(title)}</h2><div style='padding:14px;border:1px solid #dbeafe;border-radius:10px;background:#f5f7ff;'>No jobs available.</div>"
        job = item['job']
        rationale = build_reason(item, title)
        return """
        <h2 style="color:#0b3a82;">{title}</h2>
        <div style="background:#f5f7ff;border:1px solid #dbeafe;border-radius:12px;padding:16px;">
            <div style="font-size:1.25em;font-weight:700;color:#0b3a82;">{job_title}</div>
            <div style="color:#1f2937;">{location}</div>
            {salary}
            <p style="margin-top:10px;color:#374151;">{summary}</p>
            <a href="/doctor/job/{job_id}" style="background:#0b3a82;color:white;padding:10px 20px;border-radius:8px;text-decoration:none;display:inline-block;margin-top:8px;">View Job</a>
        </div>
        <p style="font-style:italic;color:#4b5563;">Why this job? {reason}</p>
        """.format(
            title=html.escape(title),
            job_title=html.escape(job.title or "Untitled Role"),
            location=html.escape(job.location or "Location TBD"),
            salary=f"<div style='color:#0b3a82;font-weight:600;margin-top:6px;'>{html.escape(job.salary or 'Salary TBD')}</div>" if job.salary else "",
            summary=html.escape((job.description or "")[:320] or "No description provided."),
            job_id=job.id,
            reason=html.escape(rationale)
        )

    def build_fallback_html():
        if not jobs:
            return "<div style='padding:18px;text-align:center;'>No job postings are available to search right now.</div>"

        sections = [
            ("Best Overall Fit", best_overall),
            ("Best Fit for Location", best_location),
            ("Best Fit for Lifestyle", best_lifestyle),
            ("Best Fit for Job-Specific Wants", best_wants),
        ]
        parts = [
            "<div style='font-family:Helvetica,Arial,sans-serif;'>",
            "<h1 style=\"color:#0b3a82;font-size:2.2em;margin-bottom:0.2em;\">Exciting Job Opportunities Tailored For You!</h1>",
            "<p style=\"color:#374151;\">We're showing locally ranked matches while AI search is unavailable.</p>",
        ]
        for title, item in sections:
            parts.append(render_card(title, item))
        parts.append("</div>")
        return "".join(parts)

    prompt = f"""
You are a professional AI assistant that matches doctors with jobs.
You must only reference the information provided in the job listings and the doctor's preferences below—do not invent or assume lifestyle, city perks, or job benefits unless explicitly present in the job description.
Your output must always use consistent, professional HTML, styled in a blue color scheme (#0066cc for headers and buttons), and avoid any images.
Show four sections:
1. The Best Overall Fit (the single job that best matches ALL criteria together).
2. The Best Fit for Location (the job most aligned with the doctor's location preference, even if it's also the overall fit).
3. The Best Fit for Lifestyle (the job most aligned with the lifestyle preference).
4. The Best Fit for Job-Specific Wants (the job most aligned with the job-specific wants).

For each section:
- Use a prominent blue header (e.g., `<h2 style=\"color:#0066cc;\">Best Overall Fit</h2>`)
- In a card-style `<div>` (light blue background: #eaf2fb, rounded corners, subtle blue border), show:
    - Job Title (big, bold, blue: #0066cc)
    - Location
    - Salary (if available)
    - Professional, vivid summary explaining why this job fits the criteria, referencing the doctor's preferences.
    - A large blue "View Job" button: `<a href=\"/doctor/job/{{job_id}}\" style=\"background:#0066cc;color:white;padding:10px 28px;font-size:1.1em;border-radius:7px;display:inline-block;text-decoration:none;margin-top:10px;\">View Job</a>`
    - Below the card, show a short "Why This Job?" blurb in italic, describing why this was selected for that category.
- Do not use code blocks.
- Do not include any images, icons, or emojis.
- If a job is selected in multiple categories, show it again in those sections (repeat is OK).
- Only pick from the jobs provided below.

Doctor's preferences:
- Lifestyle: {lifestyle}
- Job-specific wants: {wants}
- Location: {location}
- Licensed states: {', '.join(doctor_profile['licensed_states']) or 'Not provided'}
- Preferred states: {', '.join(doctor_profile['preferred_states']) or 'Not provided'}

Job list (each includes id, title, location, salary, description):
{json.dumps(jobs_payload, indent=2)}

Your response must be fully-rendered HTML, ready to be dropped into a modal.
The main title should be <h1 style=\"color:#0066cc;font-size:2.3em;margin-bottom:0.3em;\">Exciting Job Opportunities Tailored For You!</h1>
Do not output any <img> tags or links to images. Only output the requested job sections, each with consistent blue theme.
    """

    gpt_html = None
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        try:
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[
                    {"role": "system", "content": "You are a professional and creative medical job match assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2200,
                temperature=0.5,
            )
            gpt_html = response.choices[0].message.content
        except Exception as exc:
            print(f"AI search failed, using fallback: {exc}")
    else:
        print("AI search skipped: missing OPENAI_API_KEY")

    if not gpt_html:
        gpt_html = build_fallback_html()

    return jsonify({'html': gpt_html})


def get_or_create_system_user():
    """Ensure there's a system user to own automatically imported jobs."""
    system_user = User.query.filter_by(username="system_jobs").first()
    if system_user:
        return system_user

    system_user = User(
        username="system_jobs",
        email="system_jobs@example.com",
        role="admin",
    )
    system_user.set_password(os.getenv("SYSTEM_JOBS_PASSWORD", "system-jobs-placeholder"))
    db.session.add(system_user)
    db.session.commit()
    return system_user


def sync_direct_jobs_from_excel():
    """Populate or refresh jobs from JOBSDIRECTJOBS.xlsx."""
    excel_path = Path(__file__).resolve().parent / "JOBSDIRECTJOBS.xlsx"
    if not excel_path.exists():
        return

    required_columns = {"Job URL", "Job Title", "Location", "Date Posted", "Description"}
    try:
        df = pd.read_excel(excel_path)
    except Exception as exc:
        print(f"Unable to read direct jobs Excel: {exc}")
        return

    missing = required_columns - set(df.columns)
    if missing:
        print(f"Direct jobs Excel missing columns: {', '.join(sorted(missing))}")
        return

    poster = get_or_create_system_user()
    jobs_to_geocode = []
    for _, row in df.iterrows():
        job_url = str(row.get("Job URL", "") or "").strip()
        title = str(row.get("Job Title", "") or "").strip() or "Untitled Job"

        location_value = row.get("Location", "")
        location = "" if pd.isna(location_value) else str(location_value or "").strip()
        date_value = row.get("Date Posted")
        if pd.isna(date_value):
            date_posted = ""
        elif isinstance(date_value, (datetime, pd.Timestamp)):
            date_posted = date_value.strftime("%m/%d/%Y")
        else:
            date_posted = str(date_value).strip()

        description = str(row.get("Description", "") or "").strip()

        job = None
        if job_url:
            job = Job.query.filter_by(job_url=job_url).first()
        if not job and location:
            job = Job.query.filter_by(title=title, location=location).first()

        if not job:
            job = Job(
                title=title,
                location=location,
                salary="",
                description=description,
                poster_id=poster.id if poster else None,
                latitude=None,
                longitude=None,
                job_url=job_url or None,
                date_posted=date_posted or None,
            )
            db.session.add(job)
        else:
            previous_location = job.location
            job.title = title
            job.location = location
            job.description = description
            job.job_url = job_url or job.job_url
            job.date_posted = date_posted or job.date_posted

            # Clear coordinates if the location has changed so we can refresh them
            if location and previous_location and previous_location != location:
                job.latitude = None
                job.longitude = None

        # Geocode if coordinates are missing and we have a location
        if location and (job.latitude is None or job.longitude is None):
            jobs_to_geocode.append(job)

    db.session.commit()

    # Avoid long request times by limiting how many geocoding calls run per page load.
    geocoded = 0
    for job in jobs_to_geocode:
        if geocoded >= DIRECT_JOBS_GEOCODE_LIMIT:
            break

        lat, lng = geocode_location(job.location)
        job.latitude = lat
        job.longitude = lng
        geocoded += 1

    if geocoded:
        db.session.commit()
@app.route('/doctor/jobs')
@login_required
def doctor_jobs():
    if current_user.role != 'doctor':
        flash('Unauthorized access.', 'danger')
        return redirect(url_for('home'))

    sync_direct_jobs_from_excel()

    keyword = request.args.get('keyword', '').lower()
    location = request.args.get('location', '').lower()

    jobs_query = Job.query

    if keyword:
        jobs_query = jobs_query.filter(Job.title.ilike(f"%{keyword}%") | Job.description.ilike(f"%{keyword}%"))
    if location:
        jobs_query = jobs_query.filter(Job.location.ilike(f"%{location}%"))



    jobs = jobs_query.order_by(Job.id.desc()).all()
    marker_groups = defaultdict(list)
    for job in jobs:
        if job.latitude is not None and job.longitude is not None:
            key = (round(job.latitude, 5), round(job.longitude, 5))  # rounding avoids float mismatch
            marker_groups[key].append(job)

    job_markers = []
    for (lat, lng), joblist in marker_groups.items():
        job_entries = []
        for job in joblist:
            job_entries.append({
                "id": job.id,
                "title": job.title,
                "location": job.location,
            })
        job_markers.append({
            "lat": lat,
            "lng": lng,
            "jobs": job_entries
        })
    return render_template('doctor_jobs.html', jobs=jobs, job_markers=job_markers, keyword=keyword, location=location)

    

@app.route('/doctor/job/<int:job_id>', methods=['GET', 'POST'])
@login_required
def view_job(job_id):
    if current_user.role != 'doctor':
        flash('Access denied.', 'danger')
        return redirect(url_for('home'))

    job = Job.query.get_or_404(job_id)

    # Check if doctor already expressed interest
    already_interested = Message.query.filter_by(
        sender_id=current_user.id,
        job_id=job.id,
        message_type='interest'
    ).first()

    if request.method == 'POST':
        if already_interested:
            flash('You have already expressed interest in this job.', 'info')
        else:
            recipient_user = job.poster or get_or_create_system_user()
            if not recipient_user:
                flash('No recipient is available for this job.', 'warning')
            else:
                message = Message(
                    sender_id=current_user.id,
                    recipient_id=recipient_user.id,
                    job_id=job.id,
                    doctor_id=current_user.doctor.id,
                    content=f"Dr. {current_user.doctor.first_name} {current_user.doctor.last_name} expressed interest in your job '{job.title}'.",
                    message_type='interest'
                )
                db.session.add(message)
                db.session.commit()
                flash('Your interest has been sent to the client.', 'success')
        return redirect(url_for('doctor_jobs'))
    return render_template('view_job.html', job=job, already_interested=already_interested)



@app.route('/doctor/dashboard')
@login_required
def doctor_dashboard():
    if current_user.role != 'doctor':
        flash('Unauthorized access.', 'danger')
        return redirect(url_for('dashboard'))

    doctor = current_user.doctor

    # Ensure the latest direct jobs are available for suggestions and AI search.
    sync_direct_jobs_from_excel()

    scheduled_calls = ScheduledCall.query.filter_by(doctor_id=doctor.id).order_by(ScheduledCall.datetime.asc()).all()
    pending_invites = ScheduledCall.query.filter_by(doctor_id=doctor.id, invite_status='Pending').all()
    inbox_preview = (
        Message.query
        .filter_by(recipient_id=current_user.id)
        .order_by(Message.timestamp.desc())
        .limit(10)
        .all()
    )

    events = []
    for call in scheduled_calls:
        # Determine the color and status clearly
        if call.canceled:
            color, status = '#ff4d4d', 'Canceled'
        elif call.reschedule_requested:
            color, status = '#3788d8', 'Reschedule Requested'
        elif call.invite_status.lower() == 'pending':
            color, status = '#ffc107', 'Pending Invite'
        elif call.invite_status.lower() == 'accepted':
            color, status = '#28a745', 'Accepted'
        else:
            color, status = '#6c757d', 'Scheduled'

        events.append({
            'id': call.id,
            'title': f"Call with {call.scheduled_by.username}",
            'start': call.datetime.isoformat(),
            'color': color,
            'status': status
        })

    return render_template(
        'doctor_dashboard.html',
        doctor=doctor,
        events=events,
        pending_invites=pending_invites,
        inbox_preview=inbox_preview,
    )


@app.route('/doctor/suggested_jobs')
@login_required
def doctor_suggested_jobs():
    if current_user.role != 'doctor':
        return jsonify({'error': 'Unauthorized'}), 403

    doctor = current_user.doctor
    doctor_profile = {
        "name": f"Dr. {doctor.first_name or ''} {doctor.last_name or ''}".strip(),
        "specialty": doctor.specialty or "",
        "subspecialty": doctor.subspecialty or "",
        "home_base": doctor.city_of_residence or "",
        "licensed_states": normalize_state_values((doctor.states_licensed or "").split(',')),
        "preferred_states": normalize_state_values((doctor.states_willing_to_work or "").split(',')),
        "salary_expectation": doctor.salary_expectations or 0,
    }
    jobs_payload = get_doctor_jobs_payload(doctor)

    prompt = f"""
You are matching physician jobs to a doctor. Only consider the provided jobs; never invent new roles.
Pick up to 10 jobs that explicitly match the doctor's specialty or subspecialty. If a job does not match the specialty terms, do not include it.



Doctor profile:
- Name: {doctor_profile['name']}
- Specialty: {doctor_profile['specialty']}
- Subspecialty: {doctor_profile['subspecialty']}
- Home base: {doctor_profile['home_base']}
- Licensed states: {', '.join(doctor_profile['licensed_states']) or 'Not provided'}
- Preferred work states: {', '.join(doctor_profile['preferred_states']) or 'Not provided'}
- Target compensation: {doctor_profile['salary_expectation']} (numeric if provided)

Ranking rules (in order):
1) Exact specialty/subspecialty alignment (required before anything else).
2) Location fit by licensed states (highest priority) and then preferred states.
3) Compensation potential (higher salary next when available).
4) Mention of credentials or experience that aligns with the profile.


Return ONLY strict JSON (no code fences). The JSON must be an array of objects with these fields:
[
  {{
    "id": <job id>,
    "title": "<job title>",
    "location": "<city/state or location>",
    "salary": "<salary string if any>",
    "rationale": "Two concise sentences on why this is a fit, referencing specialty, pay, and location/credentials.",
    "score": <0-100 reflecting fit>
  }}
]
Sort the array by score descending and cap it at 10 items. If no jobs match the specialty, return an empty JSON array.

Jobs to evaluate (JSON):
{json.dumps(jobs_payload, indent=2)}
    """


    suggestions = []
    if jobs_payload:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("AI suggestions skipped: missing OPENAI_API_KEY; using fallback results.")
            suggestions = build_fallback_suggestions(jobs_payload, doctor_profile)
        else:
            try:
                client = openai.OpenAI(api_key=api_key)
                response = client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[
                        {"role": "system", "content": "You are a precise medical job-matching assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=1200,
                    temperature=0.2,
                )

                raw_content = response.choices[0].message.content
                parsed = json.loads(raw_content)
                if isinstance(parsed, list):
                    suggestions = parsed
            except Exception as exc:
                print(f"AI suggestions failed: {exc}; using fallback results.")
                suggestions = build_fallback_suggestions(jobs_payload, doctor_profile)

    if not suggestions and jobs_payload:
        suggestions = build_fallback_suggestions(jobs_payload, doctor_profile)

    return jsonify({"suggestions": suggestions})


@app.route('/doctor/refine_suggestions', methods=['POST'])
@login_required
def doctor_refine_suggestions():
    if current_user.role != 'doctor':
        return jsonify({'error': 'Unauthorized'}), 403

    doctor = current_user.doctor
    payload = request.get_json(silent=True) or {}
    context = (payload.get('context') or '').strip()

    doctor_profile = {
        "name": f"Dr. {doctor.first_name or ''} {doctor.last_name or ''}".strip(),
        "specialty": doctor.specialty or "",
        "subspecialty": doctor.subspecialty or "",
        "home_base": doctor.city_of_residence or "",
        "licensed_states": normalize_state_values((doctor.states_licensed or "").split(',')),
        "preferred_states": normalize_state_values((doctor.states_willing_to_work or "").split(',')),
        "salary_expectation": doctor.salary_expectations or 0,
    }

    # Always pull the full specialty-scoped job pool so refinement considers every option,
    # not just the initial shortlist the UI sent over.
    jobs = get_doctor_jobs_payload(doctor)

    if not jobs:
        return jsonify({"suggestions": [], "base": []})
    prompt = f"""
You are refining an existing shortlist of physician job matches. Only consider the provided jobs; never invent or rename roles.
The doctor shared new preferences: {context or 'No new details provided.'}

Instructions:
- Re-rank the provided jobs based on the new context while keeping specialty alignment important.
- Remove a job only if it clearly conflicts with the new information; otherwise keep it in consideration.
- Return up to 5 jobs ordered best to worst. Do not add new jobs.
- Keep the original id, title, and salary values. You may update rationale and score.

Return strictly JSON (no prose, no code fences):
[
  {{
    "id": <job id>,
    "title": "<job title>",
    "location": "<location>",
    "salary": "<salary>",
    "rationale": "Concise explanation using the new context and existing details.",
    "score": <0-100>
  }}
]
Sort by score descending and cap at 5 results. If nothing fits, return an empty array.

Jobs to refine (JSON):
{json.dumps(jobs, indent=2)}
    """

    refined = []
    try:
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You are a precise medical job-matching assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=800,
            temperature=0.2,
        )

        raw_content = response.choices[0].message.content
        parsed = json.loads(raw_content)
        if isinstance(parsed, list):
            refined = parsed
    except Exception:
        refined = build_fallback_suggestions(jobs, doctor_profile)

    top_refined = refined[:3]
    return jsonify({"suggestions": top_refined, "base": jobs})




@app.route('/handle_invite/<int:call_id>', methods=['POST'])
@login_required
def handle_invite(call_id):
    scheduled_call = ScheduledCall.query.get_or_404(call_id)

    if current_user.doctor.id != scheduled_call.doctor_id:
        flash('Unauthorized.', 'danger')
        return redirect(url_for('doctor_dashboard'))

    action = request.form.get('action')
    if action == 'accept':
        scheduled_call.invite_status = 'accepted'
        flash('Invite accepted.', 'success')

        # Notify client of acceptance
        content = f"Dr. {current_user.doctor.first_name} {current_user.doctor.last_name} accepted your call invite."
    elif action == 'decline':
        scheduled_call.invite_status = 'declined'
        flash('Invite declined.', 'info')

        # Notify client of decline
        content = f"Dr. {current_user.doctor.first_name} {current_user.doctor.last_name} declined your call invite."
    else:
        flash('Invalid action.', 'danger')
        return redirect(url_for('doctor_dashboard'))

    # Send notification message back to client
    notification = Message(
        sender_id=current_user.id,
        recipient_id=scheduled_call.scheduled_by_id,
        doctor_id=current_user.doctor.id,
        content=content,
        message_type='invite_response'
    )

    db.session.add(notification)
    db.session.commit()

    return redirect(url_for('doctor_dashboard'))


@app.route('/doctor/handle_invite/<int:call_id>', methods=['POST'])
@login_required
def doctor_handle_invite(call_id):
    scheduled_call = ScheduledCall.query.get_or_404(call_id)

    action = request.form.get('action')

    if action == 'accept':
        scheduled_call.invite_status = "Accepted"
        flash('Invite accepted!', 'success')

        notification = Message(
            sender_id=current_user.id,
            recipient_id=scheduled_call.scheduled_by_id,
            content=f"Dr. {current_user.doctor.first_name} {current_user.doctor.last_name} accepted your meeting invite scheduled on {scheduled_call.datetime.strftime('%Y-%m-%d %H:%M')}."
        )

    elif action == 'decline':
        scheduled_call.invite_status = "Declined"
        flash('Invite declined!', 'danger')

        notification = Message(
            sender_id=current_user.id,
            recipient_id=scheduled_call.scheduled_by_id,
            content=f"Dr. {current_user.doctor.first_name} {current_user.doctor.last_name} declined your meeting invite scheduled on {scheduled_call.datetime.strftime('%Y-%m-%d %H:%M')}."
        )

    db.session.add(notification)
    db.session.commit()

    return redirect(url_for('doctor_dashboard'))

@app.route('/admin/analytics')
@login_required
def admin_analytics():
    if current_user.role != 'admin':
        flash('Unauthorized access!', 'danger')
        return redirect(url_for('home'))

    jobs = Job.query.order_by(Job.id.desc()).all()

    job_data = []
    for job in jobs:
        messages = Message.query.filter_by(job_id=job.id, message_type='interest').all()

        # Count interest per day
        interest_by_day = defaultdict(int)
        interested_doctors = []
        for msg in messages:
            date_str = msg.timestamp.strftime("%Y-%m-%d")
            interest_by_day[date_str] += 1

            if msg.doctor:
                interested_doctors.append({
                    'id': msg.doctor.id,
                    'name': f"{msg.doctor.first_name} {msg.doctor.last_name}",
                    'email': msg.doctor.email
                })

        job_data.append({
            'id': job.id,
            'title': job.title,
            'location': job.location,
            'salary': job.salary,
            'description': job.description,
            'interest_count': len(messages),
            'interested_doctors': interested_doctors,
            'interest_by_day': interest_by_day,
            'client_name': job.poster.username if job.poster else "Unknown",
            'client_email': job.poster.email if job.poster else "",
        })

    return render_template("admin_analytics.html", job_data=job_data)

@app.route('/doctor/handle_reschedule/<int:call_id>', methods=['GET', 'POST'])
@login_required
def doctor_handle_reschedule(call_id):
    scheduled_call = ScheduledCall.query.get_or_404(call_id)

    if current_user.id != scheduled_call.doctor.user.id:
        flash('Unauthorized', 'danger')
        return redirect(url_for('doctor_dashboard'))

    if request.method == 'POST':
        action = request.form.get('action')

        if action == 'accept':
            scheduled_call.datetime = scheduled_call.reschedule_datetime
            scheduled_call.reschedule_requested = False
            scheduled_call.reschedule_note = None
            scheduled_call.reschedule_datetime = None
            db.session.commit()
            flash('Reschedule accepted.', 'success')

        elif action == 'reject':
            scheduled_call.reschedule_requested = False
            scheduled_call.reschedule_note = None
            scheduled_call.reschedule_datetime = None
            db.session.commit()
            flash('Reschedule request rejected.', 'warning')

        return redirect(url_for('doctor_dashboard'))

    return render_template('handle_reschedule.html', call=scheduled_call)

@app.route('/register_client', methods=['GET', 'POST'])
@login_required
def register_client():
    if current_user.role != 'admin':
        flash('Unauthorized access!', 'danger')
        return redirect(url_for('home'))

    form = ClientRegistrationForm()
    if form.validate_on_submit():
        existing_user = User.query.filter(
            (User.username == form.username.data) | (User.email == form.email.data)
        ).first()

        if existing_user:
            flash('Username or email already exists.', 'danger')
            return redirect(url_for('register_client'))

        new_client = User(
            username=form.username.data,
            email=form.email.data,
            password_hash=generate_password_hash(form.password.data),
            role='client'
        )
        db.session.add(new_client)
        db.session.commit()

        flash('New client account created successfully!', 'success')
        return redirect(url_for('home'))

    return render_template('register_client.html', form=form)

@app.route('/doctor/job/<int:job_id>/express_interest', methods=['POST'])
@login_required
def express_interest(job_id):
    job = Job.query.get_or_404(job_id)

    if current_user.role != 'doctor':
        flash('Unauthorized', 'danger')
        return redirect(url_for('doctor_jobs'))

    recipient_user = job.poster

    message = Message(
        sender_id=current_user.id,
        recipient_id=recipient_user.id,
        job_id=job.id,
        doctor_id=current_user.doctor.id,
        content=f"{current_user.doctor.first_name} {current_user.doctor.last_name} expressed interest in your job: '{job.title}'.",
        message_type='interest'  # <-- clearly marked interest message
    )

    db.session.add(message)
    db.session.commit()

    flash('Interest sent to client!', 'success')
    return redirect(url_for('doctor_jobs'))

@app.route('/send_invite/<int:doctor_id>/<int:job_id>', methods=['GET', 'POST'])
@login_required
def send_invite(doctor_id, job_id):
    if current_user.role not in ['client', 'admin']:
        flash('Unauthorized access.', 'danger')
        return redirect(url_for('home'))

    doctor = Doctor.query.get_or_404(doctor_id)
    doctor_user = User.query.filter_by(id=doctor.user_id).first_or_404()
    job = Job.query.get_or_404(job_id)

    form = ScheduledCallForm()
    form.doctor_id.choices = [(doctor.id, f"{doctor.first_name} {doctor.last_name} | {doctor.email}")]

    if form.validate_on_submit():
        scheduled_call = ScheduledCall(
            doctor_id=doctor.id,
            scheduled_by_id=current_user.id,
            job_id=job.id,
            datetime=form.datetime.data,
            reason=form.reason.data,
            invite_status='pending'
        )

        db.session.add(scheduled_call)
        db.session.commit()

        # Send invite message to doctor
        message = Message(
            sender_id=current_user.id,
            recipient_id=doctor_user.id,
            job_id=job_id,
            doctor_id=doctor.id,
            content=f"You have a call invite scheduled by {current_user.username} on {form.datetime.data}.",
            message_type='invite'
        )
        db.session.add(message)
        db.session.commit()

        flash('Invite sent to doctor!', 'success')

        # Redirect to appropriate dashboard
        if current_user.role == 'admin':
            return redirect(url_for('admin_dashboard'))
        else:
            return redirect(url_for('client_dashboard'))

    return render_template('schedule_call.html', form=form, job=job, doctor=doctor)


@app.route('/edit_job/<int:job_id>', methods=['GET', 'POST'])
@login_required
def edit_job(job_id):
    job = Job.query.get_or_404(job_id)

    if current_user.role != 'client' or job.poster_id != current_user.id:
        flash('Unauthorized access!', 'danger')
        return redirect(url_for('home'))

    form = JobForm(obj=job)

    if form.validate_on_submit():
        job.title = form.title.data
        job.location = form.location.data
        job.salary = form.salary.data
        job.description = form.description.data

        db.session.commit()
        flash('Job updated successfully!', 'success')
        return redirect(url_for('client_my_jobs'))

    return render_template('edit_job.html', form=form, job=job)


# ✅ Full route: doctor_edit_profile
@app.route('/doctor/edit_profile', methods=['GET', 'POST'])
@login_required
def doctor_edit_profile():
    doctor = current_user.doctor
    form = DoctorForm()

    if request.method == 'POST':
        if form.validate_on_submit():
            try:
                existing_doctor = Doctor.query.filter(
                    Doctor.email == form.email.data, Doctor.id != doctor.id
                ).first()
                if existing_doctor:
                    flash('Another doctor with this email already exists.', 'danger')
                    return redirect(url_for('edit_doctor', doctor_id=doctor.id))

                # ONLY handle image update if cropper input is submitted
                cropped_data = request.form.get('cropped_image_data')
                if cropped_data:
                    import base64
                    from PIL import Image
                    from io import BytesIO

                    header, encoded = cropped_data.split(",", 1)
                    binary_data = base64.b64decode(encoded)
                    img = Image.open(BytesIO(binary_data))

                    cropped_filename = f"cropped_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
                    cropped_path = os.path.join(app.static_folder, 'upload', cropped_filename)
                    img.save(cropped_path)
                    doctor.profile_picture = f"upload/{cropped_filename}"

                doctor.position = form.position.data
                doctor.specialty = form.specialty.data
                doctor.subspecialty = form.subspecialty.data
                doctor.first_name = form.first_name.data
                doctor.last_name = form.last_name.data
                doctor.email = form.email.data
                doctor.phone = form.phone.data
                doctor.alt_phone = form.alt_phone.data
                doctor.address = form.address.data
                doctor.city_of_residence = format_city_state(form.city.data, form.state.data)
                doctor.medical_school = form.medical_school.data
                doctor.med_grad_month_year = form.med_grad_month_year.data
                doctor.residency = form.residency.data
                doctor.residency_grad_month_year = form.residency_grad_month_year.data

                num_fellowships = int(form.num_fellowships.data)
                doctor.fellowship = ",".join(form.fellowship.data[:num_fellowships])
                doctor.fellowship_grad_month_year = ",".join(form.fellowship_grad_month_year.data[:num_fellowships])

                doctor.bachelors = form.bachelors.data
                doctor.bachelors_grad_month_year = form.bachelors_grad_month_year.data
                doctor.msn = form.msn.data
                doctor.msn_grad_month_year = form.msn_grad_month_year.data
                doctor.dnp = form.dnp.data
                doctor.dnp_grad_month_year = form.dnp_grad_month_year.data
                doctor.additional_training = form.additional_training.data
                doctor.sponsorship_needed = form.sponsorship_needed.data or False

                num_cases = int(form.num_malpractice_cases.data)
                malpractice_data = []
                for case_form in form.malpractice_cases.entries[:num_cases]:
                    malpractice_data.append({
                        'incident_year': case_form.form.incident_year.data,
                        'outcome': case_form.form.outcome.data,
                        'payout_amount': case_form.form.payout_amount.data or 0,
                        'case_explanation': case_form.form.case_explanation.data or ''
                    })
                doctor.malpractice_cases = json.dumps(malpractice_data)

                doctor.certification = form.certification.data
                doctor.certification_specialty_area = form.certification_specialty_area.data
                doctor.clinically_active = form.clinically_active.data
                doctor.last_clinically_active = (
                    form.last_clinically_active.data if form.clinically_active.data == 'No' else None
                )
                doctor.emr = ",".join(form.emr.data)
                doctor.languages = ",".join(form.languages.data)
                doctor.states_licensed = ",".join(form.states_licensed.data)
                doctor.states_willing_to_work = ",".join(form.states_willing_to_work.data)
                doctor.salary_expectations = form.salary_expectations.data or 0.0

                db.session.commit()
                flash('Doctor updated successfully!', 'success')
                return redirect(url_for('doctor_dashboard'))

            except Exception as e:
                db.session.rollback()
                flash(f"An error occurred: {str(e)}", "danger")

    if request.method == 'GET':
        form.position.data = doctor.position
        form.specialty.data = doctor.specialty
        form.subspecialty.data = doctor.subspecialty
        form.first_name.data = doctor.first_name
        form.last_name.data = doctor.last_name
        form.email.data = doctor.email
        form.phone.data = doctor.phone
        form.alt_phone.data = doctor.alt_phone
        form.address.data = doctor.address
        form.city.data, form.state.data = split_city_state(doctor.city_of_residence)
        form.medical_school.data = doctor.medical_school
        form.med_grad_month_year.data = doctor.med_grad_month_year
        form.residency.data = doctor.residency
        form.residency_grad_month_year.data = doctor.residency_grad_month_year

        fellowships = doctor.fellowship.split(',') if doctor.fellowship else []
        fellowship_dates = doctor.fellowship_grad_month_year.split(',') if doctor.fellowship_grad_month_year else []
        form.num_fellowships.data = str(len(fellowships))
        form.fellowship.entries.clear()
        form.fellowship_grad_month_year.entries.clear()
        for f, d in zip(fellowships, fellowship_dates):
            form.fellowship.append_entry(f)
            form.fellowship_grad_month_year.append_entry(d)
        while len(form.fellowship.entries) < int(form.fellowship.max_entries):
            form.fellowship.append_entry()
            form.fellowship_grad_month_year.append_entry()

        malpractice_cases = json.loads(doctor.malpractice_cases or '[]')
        form.num_malpractice_cases.data = str(len(malpractice_cases))
        form.malpractice_cases.entries.clear()
        for case in malpractice_cases:
            entry = form.malpractice_cases.append_entry()
            entry.incident_year.data = case.get('incident_year', '')
            entry.outcome.data = case.get('outcome', '')
            entry.payout_amount.data = case.get('payout_amount', 0.0)
            entry.case_explanation.data = case.get('case_explanation', '')
        while len(form.malpractice_cases.entries) < int(form.malpractice_cases.max_entries):
            form.malpractice_cases.append_entry()
            
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
        form.emr.data = doctor.emr.split(',') if doctor.emr else []
        form.languages.data = doctor.languages.split(',') if doctor.languages else []
        form.states_licensed.data = doctor.states_licensed.split(',') if doctor.states_licensed else []
        form.states_willing_to_work.data = doctor.states_willing_to_work.split(',') if doctor.states_willing_to_work else []
        form.salary_expectations.data = doctor.salary_expectations

    return render_template('doctor_edit_profile.html', form=form, doctor=doctor, zip=zip)





@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully.', 'success')
    return redirect(url_for('login'))

@app.route('/client/inbox')
@login_required
def client_inbox():
    if current_user.role != 'client':
        flash('Access denied.', 'danger')
        return redirect(url_for('home'))

    messages = Message.query.filter_by(recipient_id=current_user.id).order_by(Message.timestamp.desc()).all()

    unread_messages = [msg for msg in messages if not msg.read]
    for msg in unread_messages:
        msg.read = True
    db.session.commit()

    return render_template('inbox.html', messages=messages, title="Client Inbox")

@app.route('/register', methods=['GET', 'POST'])
@login_required  # Only logged-in users can create new accounts
def register():
    form = RegisterForm()
    if form.validate_on_submit():
        existing_user = User.query.filter_by(username=form.username.data).first()
        if existing_user:
            flash('Username already exists.', 'danger')
        else:
            new_user = User(username=form.username.data)
            new_user.set_password(form.password.data)
            db.session.add(new_user)
            db.session.commit()
            flash('New user registered successfully!', 'success')
            return redirect(url_for('home'))
    return render_template('register.html', form=form)

@app.route('/test_db')
def test_db():
    try:
        db.session.execute("SELECT 1")
        return "Connection successful!"
    except Exception as e:
        return f"Connection failed: {e}"

@app.route('/client/dashboard')
@login_required
def client_dashboard():
    if current_user.role != 'client':
        flash('Unauthorized access!', 'danger')
        return redirect(url_for('dashboard'))

    scheduled_calls = ScheduledCall.query.filter_by(scheduled_by_id=current_user.id).all()
    reschedule_requests = ScheduledCall.query.filter_by(
        scheduled_by_id=current_user.id, reschedule_requested=True
    ).all()

    # Upcoming calls for quick summary
    upcoming_calls = [
        call for call in scheduled_calls
        if not call.canceled and call.datetime >= datetime.utcnow()
    ]

    # Job analytics for dashboard cards
    jobs = Job.query.filter_by(poster_id=current_user.id).all()
    job_interest_summary = []
    total_interest = 0
    for job in jobs:
        interest_count = Message.query.filter_by(job_id=job.id, message_type='interest').count()
        total_interest += interest_count
        job_interest_summary.append({
            'title': job.title,
            'location': job.location,
            'interest_count': interest_count
        })

    message_preview = Message.query.filter_by(recipient_id=current_user.id).order_by(
        Message.timestamp.desc()
    ).limit(4).all()

    events = []
    for call in scheduled_calls:
        if call.canceled:
            color, status = '#dc3545', 'Canceled'
        elif call.reschedule_requested:
            color, status = '#17a2b8', 'Reschedule Requested'
        elif call.invite_status == 'Pending':
            color, status = '#ffc107', 'Pending Invite'
        else:
            color, status = '#28a745', 'Accepted'

        events.append({
            'id': call.id,
            'title': f"Call with Dr. {call.doctor.first_name} {call.doctor.last_name}",
            'start': call.datetime.isoformat(),
            'color': color,
            'status': status,
        })

    return render_template(
        'client_dashboard.html',
        events=events,
        reschedule_requests=reschedule_requests,
        upcoming_calls=upcoming_calls,
        job_interest_summary=job_interest_summary,
        message_preview=message_preview,
        total_jobs=len(jobs),
        total_interest=total_interest,
        active_calls=len(upcoming_calls)
    )


@app.route('/dashboard')
@login_required
def dashboard():
    if current_user.role == 'admin':
        return redirect(url_for('admin_dashboard'))
    elif current_user.role == 'client':
        return redirect(url_for('client_dashboard'))
    elif current_user.role == 'doctor':
        return redirect(url_for('doctor_dashboard'))
    else:
        flash('Role not recognized.', 'danger')
        return redirect(url_for('login'))

@app.route('/doctor/call/<int:call_id>', methods=['GET', 'POST'])
@login_required
def doctor_call_details(call_id):
    scheduled_call = ScheduledCall.query.get_or_404(call_id)

    if current_user.role != 'doctor' or scheduled_call.doctor_id != current_user.doctor.id:
        flash('Unauthorized', 'danger')
        return redirect(url_for('doctor_dashboard'))

    if request.method == 'POST':
        action = request.form.get('action')

        if action == 'cancel':
            scheduled_call.canceled = True
            db.session.commit()
            flash('Meeting canceled.', 'success')
            return redirect(url_for('doctor_dashboard'))

        elif action == 'reschedule':
            new_datetime = request.form.get('reschedule_datetime')
            note = request.form.get('reschedule_note')

            scheduled_call.reschedule_requested = True
            scheduled_call.reschedule_note = note
            scheduled_call.reschedule_datetime = datetime.strptime(new_datetime, '%Y-%m-%dT%H:%M')
            db.session.commit()

            # Put the notification code HERE
            message = Message(
                sender_id=current_user.id,
                recipient_id=scheduled_call.scheduled_by_id,
                content=f"Reschedule requested for call on {scheduled_call.datetime.strftime('%Y-%m-%d %H:%M')} to {scheduled_call.reschedule_datetime.strftime('%Y-%m-%d %H:%M')}. Reason: {scheduled_call.reschedule_note}"
            )
            db.session.add(message)
            db.session.commit()

            flash('Reschedule requested sent.', 'success')
            return redirect(url_for('doctor_dashboard'))

    return render_template('doctor_call_details.html', call=scheduled_call)

@app.route('/client/my_jobs')
@login_required
def client_my_jobs():
    if current_user.role != 'client':
        flash('Unauthorized access!', 'danger')
        return redirect(url_for('home'))

    keyword = request.args.get('keyword', '').lower()
    location = request.args.get('location', '').lower()

    jobs_query = Job.query.filter_by(poster_id=current_user.id)

    if keyword:
        jobs_query = jobs_query.filter(Job.title.ilike(f"%{keyword}%") | Job.description.ilike(f"%{keyword}%"))
    if location:
        jobs_query = jobs_query.filter(Job.location.ilike(f"%{location}%"))

    jobs = jobs_query.order_by(Job.id.desc()).all()

    return render_template('client_my_jobs.html', jobs=jobs, keyword=keyword, location=location)



def build_job_comparison(doctor, requirement):
    """Return a dictionary of comparison results between a doctor and job needs."""

    def to_list(value):
        if not value:
            return []
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        return [item.strip() for item in str(value).split(',') if item.strip()]

    def add_result(key, matches, explanation):
        comparison[key] = {
            'matches': matches,
            'explanation': explanation
        }

    comparison = {}

    # Direct text comparisons
    for key, doc_val, job_val, label in [
        ('position', doctor.position, requirement.position, 'provider type'),
        ('specialty', doctor.specialty, requirement.specialty, 'specialty'),
        ('subspecialty', doctor.subspecialty, requirement.subspecialty, 'subspecialty'),
        ('certification', doctor.certification, requirement.certification, 'certification'),
        ('certification_specialty_area', doctor.certification_specialty_area, requirement.certification_specialty_area, 'certification focus'),
        ('clinically_active', doctor.clinically_active, requirement.clinically_active, 'clinical activity')
    ]:
        if not job_val:
            add_result(key, True, 'No job preference set for this item.')
        elif (doc_val or '').strip().lower() == (job_val or '').strip().lower():
            add_result(key, True, 'Matches job need.')
        else:
            add_result(key, False, f"Job needs {job_val}, but doctor lists {doc_val or 'no data'} for {label}.")

    # EMR comparison
    job_emrs = to_list(requirement.emr)
    doctor_emrs = to_list(doctor.emr)
    if job_emrs:
        missing_emrs = [emr for emr in job_emrs if emr.lower() not in [d.lower() for d in doctor_emrs]]
        if missing_emrs:
            add_result('emr', False, f"Missing EMR experience with {', '.join(missing_emrs)}.")
        else:
            add_result('emr', True, 'Doctor has all requested EMR experience.')
    else:
        add_result('emr', True, 'No EMR preference set for this job.')

    # Language comparison
    job_languages = to_list(requirement.languages)
    doctor_languages = to_list(doctor.languages)
    if job_languages:
        missing_languages = [lang for lang in job_languages if lang.lower() not in [d.lower() for d in doctor_languages]]
        if missing_languages:
            add_result('languages', False, f"Job prefers {', '.join(job_languages)}, doctor lists {doctor.languages or 'no languages'}.")
        else:
            add_result('languages', True, 'Doctor meets language needs.')
    else:
        add_result('languages', True, 'No language preference set for this job.')

    # State licensure comparison
    required_states = to_list(requirement.states_required)
    doctor_states = to_list(doctor.states_licensed)
    if required_states:
        missing_states = [state for state in required_states if state not in doctor_states]
        if missing_states:
            add_result('states_required', False, f"Requires licenses in {', '.join(required_states)}, doctor missing {', '.join(missing_states)}.")
        else:
            add_result('states_required', True, 'Doctor holds all required state licenses.')
    else:
        add_result('states_required', True, 'No required states specified for the job.')

    preferred_states = to_list(requirement.states_preferred)
    willing_states = to_list(doctor.states_willing_to_work)
    if preferred_states:
        missing_pref = [state for state in preferred_states if state not in willing_states]
        if missing_pref:
            add_result('states_preferred', False, f"Prefers availability in {', '.join(preferred_states)}, doctor not available in {', '.join(missing_pref)}.")
        else:
            add_result('states_preferred', True, 'Doctor open to all preferred states.')
    else:
        add_result('states_preferred', True, 'No preferred states specified for the job.')

    # Sponsorship support vs need
    if requirement.sponsorship_supported:
        add_result('sponsorship', True, 'Job supports sponsorship needs.')
    else:
        if doctor.sponsorship_needed:
            add_result('sponsorship', False, 'Job does not support sponsorship but doctor requires it.')
        else:
            add_result('sponsorship', True, 'No sponsorship needed for this doctor.')

    # Salary alignment
    def parse_salary_range(range_text):
        if not range_text:
            return None, None
        numbers = re.findall(r"\d+", range_text.replace(',', ''))
        if not numbers:
            return None, None
        if len(numbers) == 1:
            value = float(numbers[0])
            return value, value
        return float(numbers[0]), float(numbers[1])

    min_salary, max_salary = parse_salary_range(requirement.salary_range)
    if doctor.salary_expectations and max_salary:
        if doctor.salary_expectations > max_salary:
            add_result('salary', False, f"Doctor expects ${doctor.salary_expectations:,.0f}, above job budget of ${max_salary:,.0f}.")
        elif min_salary and doctor.salary_expectations < min_salary:
            add_result('salary', False, f"Doctor expects ${doctor.salary_expectations:,.0f}, below stated range starting at ${min_salary:,.0f}.")
        else:
            add_result('salary', True, 'Doctor salary expectations align with job range.')
    elif max_salary:
        add_result('salary', True, 'Job lists salary range; doctor did not specify expectations.')
    else:
        add_result('salary', True, 'No salary range specified for this job.')

    return comparison


@app.route('/doctor/<int:doctor_id>')
def doctor_profile(doctor_id):
    doctor = Doctor.query.get_or_404(doctor_id)
    malpractice_cases = json.loads(doctor.malpractice_cases or '[]')
    job_id = request.args.get('job_id', type=int)
    compare_mode = request.args.get('compare') == '1'

    job_requirement = None
    associated_job = None
    comparison = {}

    if job_id:
        associated_job = Job.query.get(job_id)
        if associated_job:
            job_requirement = associated_job.requirements

    if compare_mode and job_requirement:
        comparison = build_job_comparison(doctor, job_requirement)

    return render_template(
        'doctor_profile.html',
        doctor=doctor,
        malpractice_cases=malpractice_cases,
        comparison=comparison,
        compare_mode=compare_mode,
        job_requirement=job_requirement,
        job_id=job_id,
        associated_job=associated_job
    )
@app.route('/client/handle_reschedule/<int:call_id>', methods=['POST'])
@login_required
def client_handle_reschedule(call_id):
    scheduled_call = ScheduledCall.query.get_or_404(call_id)
    action = request.form.get('action')
    client_note = request.form.get('client_note', '')

    if action == 'accept':
        scheduled_call.datetime = scheduled_call.reschedule_datetime
        scheduled_call.reschedule_requested = False
        scheduled_call.reschedule_note = None
        scheduled_call.reschedule_datetime = None
        content = (f"Your reschedule request for "
                   f"{scheduled_call.datetime.strftime('%Y-%m-%d %H:%M')} has been accepted. {client_note}")
    elif action == 'decline':
        scheduled_call.reschedule_requested = False
        declined_datetime = scheduled_call.reschedule_datetime
        scheduled_call.reschedule_datetime = None
        scheduled_call.reschedule_note = None
        content = (f"Your reschedule request for "
                   f"{declined_datetime.strftime('%Y-%m-%d %H:%M')} has been declined. {client_note}")

    db.session.commit()

    message = Message(
        sender_id=current_user.id,
        recipient_id=scheduled_call.doctor.user_id,  # Corrected this line
        content=content,
        timestamp=datetime.utcnow()
    )

    db.session.add(message)
    db.session.commit()

    flash('Reschedule handled successfully.', 'success')
    return redirect(url_for('client_dashboard'))


@app.route('/client/analytics/download/<int:job_id>')
@login_required
def download_job_applicants(job_id):
    job = Job.query.get_or_404(job_id)
    # Allow if admin OR the client who posted the job
    if current_user.role not in ['client', 'admin']:
        flash('Unauthorized access!', 'danger')
        return redirect(url_for('home'))
    if current_user.role == 'client' and job.poster_id != current_user.id:
        flash('Not authorized for this job.', 'danger')
        return redirect(url_for('client_analytics'))

    # Get all doctors who expressed interest (via Message table)
    messages = Message.query.filter_by(job_id=job.id, message_type='interest').all()
    doctor_ids = {msg.doctor_id for msg in messages if msg.doctor_id}

    # Get Doctor objects
    doctors = Doctor.query.filter(Doctor.id.in_(doctor_ids)).all()

    # Prepare data for Excel (all fields in Doctor model)
    rows = []
    for doc in doctors:
        row = {
            "First Name": doc.first_name,
            "Last Name": doc.last_name,
            "Healthcare Provider Type": doc.position,
            "Specialty": doc.specialty,
            "Subspecialty": doc.subspecialty,
            "Email": doc.email,
            "Phone": doc.phone,
            "Alt Phone": doc.alt_phone,
            "City of Residence": doc.city_of_residence,
            "Medical School": doc.medical_school,
            "Medical School Graduation (Month/Year)": doc.med_grad_month_year,
            "Residency": doc.residency,
            "Residency Grad Month/Year": doc.residency_grad_month_year,
            "Fellowship(s)": doc.fellowship,
            "Fellowship Grad Month/Year": doc.fellowship_grad_month_year,
            "Bachelors": doc.bachelors,
            "Bachelors Grad Month/Year": doc.bachelors_grad_month_year,
            "MSN": doc.msn,
            "MSN Grad Month/Year": doc.msn_grad_month_year,
            "DNP": doc.dnp,
            "DNP Grad Month/Year": doc.dnp_grad_month_year,
            "Additional Training": doc.additional_training,
            "Sponsorship Needed": "Yes" if doc.sponsorship_needed else "No",
            "Certification": doc.certification,
            "Certification Specialty": doc.certification_specialty_area,
            "Clinically Active": doc.clinically_active,
            "Last Clinically Active": doc.last_clinically_active,
            "EMR": doc.emr,
            "Languages": doc.languages,
            "States Licensed": doc.states_licensed,
            "States Willing to Work": doc.states_willing_to_work,
            "Salary Expectation (Total Compensation)": doc.salary_expectations,
            "Joined": doc.joined.strftime('%Y-%m-%d') if doc.joined else "",
            # You can add more fields here if needed!
        }
        # Optionally, include Malpractice cases (as JSON string)
        row["Malpractice Cases"] = doc.malpractice_cases
        rows.append(row)

    # Generate Excel
    df = pd.DataFrame(rows)
    output = BytesIO()
    df.to_excel(output, index=False)
    output.seek(0)

    filename = f"Applicants_{job.title.replace(' ','_')}.xlsx"
    return send_file(
        output,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        download_name=filename,
        as_attachment=True
    )

@app.route('/edit_doctor/<int:doctor_id>', methods=['GET', 'POST'])
@login_required
def edit_doctor(doctor_id):
    doctor = Doctor.query.get_or_404(doctor_id)
    form = DoctorForm()

    if form.validate_on_submit():
        try:
            existing_doctor = Doctor.query.filter(
                Doctor.email == form.email.data, Doctor.id != doctor_id
            ).first()
            if existing_doctor:
                flash('Another doctor with this email already exists.', 'danger')
                return redirect(url_for('edit_doctor', doctor_id=doctor_id))

            doctor.position = form.position.data
            doctor.specialty = form.specialty.data
            doctor.subspecialty = form.subspecialty.data
            doctor.first_name = form.first_name.data
            doctor.last_name = form.last_name.data
            doctor.email = form.email.data
            doctor.phone = form.phone.data
            doctor.alt_phone = form.alt_phone.data
            doctor.address = form.address.data
            doctor.city_of_residence = format_city_state(form.city.data, form.state.data)
            doctor.medical_school = form.medical_school.data
            doctor.med_grad_month_year = form.med_grad_month_year.data
            doctor.residency = form.residency.data
            doctor.residency_grad_month_year = form.residency_grad_month_year.data

            # Handle Fellowships dynamically
            num_fellowships = int(form.num_fellowships.data or 0)
            doctor.fellowship = ",".join(form.fellowship.data[:num_fellowships])
            doctor.fellowship_grad_month_year = ",".join(
                form.fellowship_grad_month_year.data[:num_fellowships]
            )

            doctor.bachelors = form.bachelors.data
            doctor.bachelors_grad_month_year = form.bachelors_grad_month_year.data
            doctor.msn = form.msn.data
            doctor.msn_grad_month_year = form.msn_grad_month_year.data
            doctor.dnp = form.dnp.data
            doctor.dnp_grad_month_year = form.dnp_grad_month_year.data
            doctor.additional_training = form.additional_training.data
            doctor.sponsorship_needed = form.sponsorship_needed.data or False

            # Handle Malpractice Cases dynamically
            num_cases = int(form.num_malpractice_cases.data or 0)
            malpractice_data = []
            for case_form in form.malpractice_cases.entries[:num_cases]:
                malpractice_data.append({
                    'incident_year': case_form.form.incident_year.data,
                    'outcome': case_form.form.outcome.data,
                    'payout_amount': case_form.form.payout_amount.data or 0,
                    'case_explanation': case_form.form.case_explanation.data or ''
                })
            doctor.malpractice_cases = json.dumps(malpractice_data)

            doctor.certification = form.certification.data
            doctor.certification_specialty_area = form.certification_specialty_area.data
            doctor.clinically_active = form.clinically_active.data

            if form.clinically_active.data == 'No':
                doctor.last_clinically_active = form.last_clinically_active.data
            else:
                doctor.last_clinically_active = None
            emr_selections = list(form.emr.data or [])
            if form.emr_other.data:
                emr_selections.extend([item.strip() for item in form.emr_other.data.split(',') if item.strip()])

            language_selections = list(form.languages.data or [])
            if form.language_other.data:
                language_selections.extend([item.strip() for item in form.language_other.data.split(',') if item.strip()])

            doctor.emr = ",".join(emr_selections)
            doctor.languages = ",".join(language_selections)
            doctor.states_licensed = ",".join(form.states_licensed.data or [])
            doctor.states_willing_to_work = ",".join(form.states_willing_to_work.data or [])
            doctor.salary_expectations = form.salary_expectations.data or 0.0
            db.session.commit()
            flash('Doctor updated successfully!', 'success')
            return redirect(url_for('doctor_profile', doctor_id=doctor.id))

        except Exception as e:
            db.session.rollback()
            flash(f"An error occurred: {str(e)}", "danger")

    if request.method == 'GET':
    # Manually set simple fields
        form.position.data = doctor.position
        form.specialty.data = doctor.specialty
        form.subspecialty.data = doctor.subspecialty
        form.first_name.data = doctor.first_name
        form.last_name.data = doctor.last_name
        form.email.data = doctor.email
        form.phone.data = doctor.phone
        form.alt_phone.data = doctor.alt_phone
        form.address.data = doctor.address
        form.city.data, form.state.data = split_city_state(doctor.city_of_residence)
        form.medical_school.data = doctor.medical_school
        form.med_grad_month_year.data = doctor.med_grad_month_year
        form.residency.data = doctor.residency
        form.residency_grad_month_year.data = doctor.residency_grad_month_year

        # Fellowships (with limit)
        form.fellowship.entries.clear()
        form.fellowship_grad_month_year.entries.clear()
        fellowships = doctor.fellowship.split(',') if doctor.fellowship else []
        fellowship_dates = doctor.fellowship_grad_month_year.split(',') if doctor.fellowship_grad_month_year else []
        num_fellowships = min(len(fellowships), int(form.fellowship.max_entries))
        form.num_fellowships.data = str(num_fellowships)
        for i in range(num_fellowships):
            fellowship_entry = form.fellowship.append_entry()
            fellowship_entry.data = fellowships[i]
            date_entry = form.fellowship_grad_month_year.append_entry()
            date_entry.data = fellowship_dates[i] if i < len(fellowship_dates) else ''
        while len(form.fellowship.entries) < int(form.fellowship.max_entries):
            form.fellowship.append_entry()
            form.fellowship_grad_month_year.append_entry()

        # Malpractice cases (with limit)
        form.malpractice_cases.entries.clear()
        malpractice_cases = json.loads(doctor.malpractice_cases or '[]')
        num_cases = min(len(malpractice_cases), int(form.malpractice_cases.max_entries))
        form.num_malpractice_cases.data = str(num_cases)
        for case in malpractice_cases[:num_cases]:
            entry = form.malpractice_cases.append_entry()
            entry.incident_year.data = case.get('incident_year', '')
            entry.outcome.data = case.get('outcome', '')
            entry.payout_amount.data = case.get('payout_amount', 0.0)
            entry.case_explanation.data = case.get('case_explanation', '')
        while len(form.malpractice_cases.entries) < int(form.malpractice_cases.max_entries):
            form.malpractice_cases.append_entry()
        
        # NP/PA fields
        form.bachelors.data = doctor.bachelors
        form.bachelors_grad_month_year.data = doctor.bachelors_grad_month_year
        form.msn.data = doctor.msn
        form.msn_grad_month_year.data = doctor.msn_grad_month_year
        form.dnp.data = doctor.dnp
        form.dnp_grad_month_year.data = doctor.dnp_grad_month_year
        form.additional_training.data = doctor.additional_training
        form.sponsorship_needed.data = doctor.sponsorship_needed

        # Certification and additional fields
        form.certification.data = doctor.certification
        form.certification_specialty_area.data = doctor.certification_specialty_area
        form.clinically_active.data = doctor.clinically_active
        form.last_clinically_active.data = doctor.last_clinically_active
        emr_values = doctor.emr.split(',') if doctor.emr else []
        valid_emr_options = {choice for choice, _ in form.emr.choices}
        form.emr.data = [value for value in emr_values if value in valid_emr_options]
        form.emr_other.data = ", ".join([value for value in emr_values if value not in valid_emr_options])

        language_values = doctor.languages.split(',') if doctor.languages else []
        valid_language_options = {choice for choice, _ in form.languages.choices}
        form.languages.data = [value for value in language_values if value in valid_language_options]
        form.language_other.data = ", ".join([value for value in language_values if value not in valid_language_options])
        form.states_licensed.data = doctor.states_licensed.split(',') if doctor.states_licensed else []
        form.states_willing_to_work.data = doctor.states_willing_to_work.split(',') if doctor.states_willing_to_work else []
        form.salary_expectations.data = doctor.salary_expectations
            
    return render_template('edit_doctor.html', form=form, doctor=doctor, zip=zip)



with app.app_context():
    db.create_all()  # <-- Create tables first!

    # Check if admin already exists to avoid duplicates
    existing_admin = User.query.filter_by(username='adminchan').first()

    if not existing_admin:
        admin = User(
            username='adminchan',
            email='admin@example.com',
            password_hash=generate_password_hash('icecream2'),
            role='admin'
        )
        db.session.add(admin)
        db.session.commit()
        print("Admin user 'adminchan' created successfully!")
    else:
        print("Admin user already exists.")
# App initialization
threading.Thread(target=lambda: (time.sleep(1), webbrowser.open('http://localhost:5000'))).start()
@app.route('/')
def landing_page():
    return render_template('landing_page.html', current_year=datetime.now().year)

@app.route('/public_register_client', methods=['POST'])
def public_register_client():
    data = request.form

    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    confirm_password = data.get('confirm_password')

    if password != confirm_password:
        flash('Passwords do not match.', 'danger')
        return redirect(url_for('create_account'))

    if User.query.filter((User.username == username) | (User.email == email)).first():
        flash('Username or email already exists.', 'danger')
        return redirect(url_for('create_account'))

    user = User(username=username, email=email, role='client')
    user.set_password(password)
    db.session.add(user)
    db.session.commit()

    login_user(user)
    flash('Client account created and logged in!', 'success')
    return redirect(url_for('client_dashboard'))

@app.route('/client/analytics')
@login_required
def client_analytics():
    if current_user.role != 'client':
        flash('Unauthorized access!', 'danger')
        return redirect(url_for('home'))

    jobs = Job.query.filter_by(poster_id=current_user.id).order_by(Job.id.desc()).all()

    job_data = []
    for job in jobs:
        messages = Message.query.filter_by(job_id=job.id, message_type='interest').all()

        # Count interest per day
        interest_by_day = defaultdict(int)
        interested_doctors = []
        for msg in messages:
            date_str = msg.timestamp.strftime("%Y-%m-%d")
            interest_by_day[date_str] += 1

            if msg.doctor:
                interested_doctors.append({
                    'id': msg.doctor.id,
                    'name': f"{msg.doctor.first_name} {msg.doctor.last_name}",
                    'email': msg.doctor.email
                })

        job_data.append({
            'id': job.id,
            'title': job.title,
            'location': job.location,
            'salary': job.salary,
            'description': job.description,
            'interest_count': len(messages),
            'interested_doctors': interested_doctors,
            'interest_by_day': interest_by_day
        })

    return render_template("client_analytics.html", job_data=job_data)

@app.route('/public_register_doctor', methods=['POST'])
def public_register_doctor():
    data = request.form

    # Extract fields
    username = (data.get('username') or '').strip()
    email = (data.get('email') or '').strip().lower()
    password = data.get('password')
    confirm_password = data.get('confirm_password')
    first_name = data.get('first_name')
    last_name = data.get('last_name')
    specialty = data.get('specialty')
    
    # Validation
    if password != confirm_password:
        flash('Passwords do not match.', 'danger')
        return redirect(url_for('create_account'))

    if User.query.filter_by(username=username).first():
        flash('Username already exists.', 'danger')
        return redirect(url_for('create_account'))

    if User.query.filter_by(email=email).first():
        flash('This email is already tied to an account.', 'danger')
        return redirect(url_for('create_account'))

    if Doctor.query.filter_by(email=email).first():
        flash('A doctor with this email already exists.', 'danger')
        return redirect(url_for('create_account'))

    # Create user and doctor records
    user = User(username=username, email=email, role='doctor')
    user.set_password(password)
    db.session.add(user)
    db.session.flush()

    doctor = Doctor(
        first_name=first_name,
        last_name=last_name,
        email=email,
        specialty=specialty,
        position='MD',  # or allow selection if you want
        user_id=user.id,
        joined=datetime.utcnow()
    )
    db.session.add(doctor)
    db.session.commit()
    login_user(user)
    flash('Account created and logged in!', 'success')
    return redirect(url_for('doctor_dashboard'))
@app.route('/home')
def home():
    return redirect(url_for('dashboard'))

@app.route('/reset_db')
def reset_db():
    with app.app_context():
        db.drop_all()
        db.create_all()

        # Recreate your default admin user
        admin = User(
            username='adminchan',
            email='admin@example.com',
            password_hash=generate_password_hash('icecream2'),
            role='admin'
        )
        db.session.add(admin)
        db.session.commit()

    return "✅ Database has been reset and admin user created!"

with app.app_context():
    db.create_all()

    doctors = Doctor.query.all()
    print("Existing Doctors and Emails:")
    for doc in doctors:
        print(doc.first_name, doc.last_name, doc.email)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DocIV application entrypoint")
    parser.add_argument(
        "--geocode-missing-jobs",
        action="store_true",
        help="Backfill latitude/longitude for jobs missing coordinates and exit.",
    )
    args = parser.parse_args()

    if args.geocode_missing_jobs:
        geocode_missing_jobs()
    else:
        app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

































































































