from django.db import models


class Student(models.Model):
    pass


class Course(models.Model):
    pass


class Enrolment(models.Model):
    student = models.ForeignKey(Student, on_delete=models.CASCADE)
    course = models.ForeignKey(Course, on_delete=models.CASCADE)

