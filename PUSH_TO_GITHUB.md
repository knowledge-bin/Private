# How to Push to Private GitHub

## Step 1: Create Private Repository on GitHub

1. Go to https://github.com
2. Click "+" → "New repository"
3. Repository name: `profile-ablation`
4. **Important**: Select **Private**
5. Do NOT initialize with README (we have one)
6. Click "Create repository"

## Step 2: Initialize Git

```bash
cd profile-ablation-clean
git init
git add .
git commit -m "Initial commit: PROFILE ablation study framework"
```

## Step 3: Connect to GitHub

```bash
# Replace YOUR_USERNAME with your GitHub username
git remote add origin https://github.com/YOUR_USERNAME/profile-ablation.git

# Or use SSH (recommended)
git remote add origin git@github.com:YOUR_USERNAME/profile-ablation.git
```

## Step 4: Push to GitHub

```bash
# Push to main branch
git branch -M main
git push -u origin main
```

## Step 5: Clone on GPU Server

```bash
# SSH into your GPU server
ssh user@gpu-server

# Clone the repository
git clone https://github.com/YOUR_USERNAME/profile-ablation.git
cd profile-ablation

# Follow setup instructions in README.md
./setup_gpu_environment.sh
```

## Optional: Add Collaborators

If you want to give access to team members:

1. Go to repository on GitHub
2. Click "Settings" → "Collaborators"
3. Add collaborators by username/email

## Git Commands for Updates

```bash
# On your local machine (after making changes)
git add .
git commit -m "Description of changes"
git push

# On GPU server (to get updates)
git pull
```

## Verification

After pushing, verify on GitHub:
1. Go to https://github.com/YOUR_USERNAME/profile-ablation
2. Check that all files are present
3. Verify repository is marked "Private"
4. README.md should be displayed on main page

---

**Ready to push!** 🚀
