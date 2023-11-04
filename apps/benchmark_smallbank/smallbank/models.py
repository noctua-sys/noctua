from django.db import models

# Create your models here.
class Account(models.Model):
    name = models.TextField(unique=True)
    savings_bal = models.FloatField()
    checking_bal = models.FloatField()
