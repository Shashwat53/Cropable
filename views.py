from django.shortcuts import render, redirect
from django.http import HttpResponse, HttpResponseRedirect
from django.contrib.auth import login, logout, authenticate
from .forms import UserProfileForm, UserForm, LoginForm
from django.contrib.auth.decorators import login_required
from django.urls import reverse

from .forms import UserForm, UserProfileForm

# Create your views here.
def user_signup(request):
    registered = False

    if request.method == "POST":
        user_profile_form = UserProfileForm(request.POST)
        user_form = UserForm(request.POST)

        if user_profile_form.is_valid() and user_form.is_valid():
            user = user_form.save()
            user.set_password(user.password)
            user.save()

            user_profile = user_profile_form.save(commit=False)
            user_profile.user = user
            user_profile.save()

            registered = True
            return redirect("home")

        else:
            print(user_profile_form.errors)
            print(user_form.errors)

    else:
        user_profile_form = UserProfileForm()
        user_form = UserForm()

    context_dict = {
        "user_profile_form": user_profile_form,
        "user_form": user_form,
        "registered": registered,
    }

    return render(request, "signup.html", context_dict)

def user_login(request):

    if request.method == "POST":
        form = LoginForm(request.POST)

        if form.is_valid():
            user = authenticate(
                username=form.cleaned_data["username"],
                password=form.cleaned_data["password"],
            )

            if user:
                if user.is_active:
                    login(request, user)
                    return redirect("home")
                else:
                    return HttpResponse("User not active!")

            else:
                return HttpResponse("User does not exist!")

    else:
        form = LoginForm()
        return render(request, "login.html", {"form": form})


@login_required
def user_logout(request):
    logout(request)
    return HttpResponseRedirect(reverse("home"))