from django.db import models
from django.contrib.auth import get_user_model

User = get_user_model()

# Create your models here.
class UserProfile(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    phone = models.PositiveIntegerField()

    def __str__(self):
        return self.user.username