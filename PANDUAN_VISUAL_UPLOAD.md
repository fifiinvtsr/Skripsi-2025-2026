# 🎯 PANDUAN VISUAL: Upload Folder PrediksiAPI

## 📊 Diagram Alur Upload

```
┌─────────────────────────────────────────────────────────────────┐
│                    PROSES UPLOAD KE GITHUB                      │
└─────────────────────────────────────────────────────────────────┘

TAHAP 1: PERSIAPAN
─────────────────
    Folder PrediksiAPI         Repository GitHub
    (Lokal di PC)              (Cloud)
         ↓                           ↓
    C:\Users\...\              https://github.com/
    PrediksiAPI/               fifiinvtsr/
                              Skripsi-2025-2026


TAHAP 2: CLONE REPOSITORY
─────────────────────────
    GitHub                         PC Lokal
       │                              ↓
       ├─── git clone ────────────→ Skripsi-2025-2026/
       │                           (folder kosong)
       └─── .git folder ──────────→ (metadata repo)


TAHAP 3: COPY FOLDER
────────────────────
    PrediksiAPI/                 Skripsi-2025-2026/
    (Original)                   (Repository)
       │                              ↓
       ├─ PrediksiAPI/            ├─ PrediksiAPI/
       ├─ __pycache__/ ✗ (skip)   ├─ .git/
       ├─ venv/ ✗ (skip)          └─ ...
       └─ ...

    (Exclude: __pycache__, venv, .venv)


TAHAP 4: STAGING (git add)
──────────────────────────
    PrediksiAPI/
    (Local folder)
         ↓
      git add PrediksiAPI/
         ↓
    Staging Area (Index)
    ├─ app.py
    ├─ requirements.txt
    ├─ model_prediksi/
    ├─ grafik_*.png
    └─ ... (30 files)


TAHAP 5: COMMIT
───────────────
    Staging Area
         ↓
      git commit
         ↓
    Local Repository
    └─ HEAD → main
       └─ commit a6cd28b
          "Add PrediksiAPI project..."


TAHAP 6: PUSH KE GITHUB
──────────────────────
    Local Repository          GitHub Repository
    (main branch)             (remote/main)
         ↓                         ↓
      git push ──────────────→ origin/main
                              ✓ Files uploaded!


TAHAP 7: VERIFIKASI
───────────────────
    GitHub Web Interface
    https://github.com/fifiinvtsr/Skripsi-2025-2026
         ↓
    ├─ PrediksiAPI/ ✓
    │  ├─ PrediksiAPI/
    │  │  ├─ app.py
    │  │  ├─ model_prediksi/
    │  │  └─ ... (30 files)
    │  └─ UPLOAD_GUIDE.md
    └─ RINGKASAN_UPLOAD.md
```

---

## 📋 Struktur Folder Hasil Upload

```
Skripsi-2025-2026 (GitHub Repository)
│
├── 📁 PrediksiAPI/                    (Main Project Folder)
│   │
│   ├── 📁 PrediksiAPI/                (Source Code Folder)
│   │   │
│   │   ├── 🐍 app.py                 (Flask API - 86 KB)
│   │   │   └── Routes: /predict, /evaluate, /health
│   │   │
│   │   ├── 📝 requirements.txt        (Dependencies)
│   │   │   └── flask, scikit-learn, pandas, numpy, etc
│   │   │
│   │   ├── 🧪 test_hyperparameter.py (Testing Scripts)
│   │   ├── 🧪 test_per_kategori.py
│   │   │
│   │   ├── 📊 Grafik (17 PNG Files)
│   │   │   ├── grafik_6_hyperparameter.png (406 KB)
│   │   │   ├── grafik_alpha.png
│   │   │   ├── grafik_gamma.png
│   │   │   ├── grafik_lambda.png
│   │   │   ├── grafik_learning_rate.png
│   │   │   ├── grafik_max_depth.png
│   │   │   ├── grafik_n_estimators.png
│   │   │   ├── grafik_eval_*.png (5 files)
│   │   │   └── grafik_kategori_*.png (3 files)
│   │   │
│   │   ├── 📊 hasil_pengujian.xlsx    (Test Results)
│   │   │
│   │   └── 🤖 model_prediksi/        (ML Models - 608 KB)
│   │       ├── complete_mlp_sla_model.pkl
│   │       ├── final_svr_model.pkl
│   │       ├── final_svr_model_new.pkl
│   │       ├── final_svr_model_parameter_lama.pkl
│   │       ├── encoder_svr_new.pkl
│   │       ├── encoder_svr_parameter_lama.pkl
│   │       ├── scaler_svr_new.pkl
│   │       └── scaler_svr_parameter_lama.pkl
│   │
│   └── 📖 UPLOAD_GUIDE.md             (Upload Documentation)
│
├── 📖 RINGKASAN_UPLOAD.md             (Summary Document)
└── 📖 README.md (opsional)            (Project Documentation)
```

---

## 🔄 Siklus Kerja Git

```
┌─────────────────────────────────────────────────────────────┐
│              GIT WORKFLOW (Git Workflow Cycle)              │
└─────────────────────────────────────────────────────────────┘

1. MODIFY FILES (Edit / Create / Delete)
   └─ Working Directory
      └─ Status: MODIFIED ❌


2. STAGE CHANGES (git add)
   ├─ Working Directory
   │  └─ Status: MODIFIED ❌
   └─ Staging Area (Index)
      └─ Status: STAGED ⏳


3. COMMIT (git commit)
   ├─ Working Directory
   │  └─ Status: UNMODIFIED ✓
   ├─ Staging Area
   │  └─ Status: COMMITTED ✓
   └─ Local Repository
      └─ HEAD → main (a6cd28b)


4. PUSH (git push)
   ├─ Local Repository
   │  └─ main (a6cd28b)
   └─ Remote Repository
      └─ origin/main (a6cd28b) ✓


5. PULL (git pull) - untuk update dari orang lain
   ├─ Remote Repository
   │  └─ origin/main
   └─ Local Repository & Working Directory
      └─ Synchronized ✓
```

---

## 📊 Status Files di Each Stage

```
┌──────────────┬─────────────┬──────────────┬───────────────┐
│ Stage        │ Location    │ Status       │ git command   │
├──────────────┼─────────────┼──────────────┼───────────────┤
│ 1. Edit      │ Working Dir │ MODIFIED ❌  │ (edit file)   │
│ 2. Stage     │ Index       │ STAGED ⏳     │ git add       │
│ 3. Commit    │ Repository  │ COMMITTED ✓  │ git commit    │
│ 4. Push      │ Remote      │ SYNCED ✓✓    │ git push      │
└──────────────┴─────────────┴──────────────┴───────────────┘
```

---

## 🎯 Quick Reference Commands

```
┌─────────────────────────────────────────────────────────────┐
│              COMMAND QUICK REFERENCE                        │
└─────────────────────────────────────────────────────────────┘

SETUP
─────
git clone <url>                    # Clone repository
git config user.name "Name"        # Setup name
git config user.email "email"      # Setup email

CHECK STATUS
────────────
git status                         # Show changes
git log                            # Show commit history
git diff                           # Show changes detail

STAGING & COMMITTING
────────────────────
git add <file>                     # Stage file
git add .                          # Stage all files
git add <folder>/                  # Stage folder
git commit -m "message"            # Create commit
git commit -am "message"           # Stage + Commit

PUSHING & PULLING
─────────────────
git push origin main               # Push to GitHub
git pull origin main               # Pull from GitHub
git fetch origin                   # Fetch updates

BRANCHING
─────────
git branch                         # List branches
git branch <name>                  # Create branch
git checkout <branch>              # Switch branch
git checkout -b <branch>           # Create + Switch

UNDO CHANGES
────────────
git restore <file>                 # Undo changes
git reset HEAD <file>              # Unstage file
git revert HEAD                    # Revert last commit
```

---

## 📈 Statistics Upload

```
╔══════════════════════════════════════════════════════════╗
║           UPLOAD STATISTICS - PrediksiAPI                ║
╠══════════════════════════════════════════════════════════╣
║ Total Files Uploaded        : 30 files                   ║
║ Total Size                  : 3.33 MB                    ║
║ Commit Hash                 : a6cd28b                    ║
║ Commit Date                 : 10 Mei 2026, 21:34        ║
║ Branch                      : main                       ║
║ Repository                  : Skripsi-2025-2026          ║
║ Owner                       : fifiinvtsr                 ║
║ Visibility                  : Public (settable)          ║
║                                                          ║
║ FILE BREAKDOWN:                                          ║
║ ├─ Python Code              : 3 files (97 KB)           ║
║ ├─ Configuration            : 1 file (85 bytes)         ║
║ ├─ Graphs/Charts            : 17 files (2.6 MB)         ║
║ ├─ ML Models                : 8 files (608 KB)          ║
║ └─ Spreadsheets             : 1 file (15 KB)            ║
║                                                          ║
║ SIZE BREAKDOWN:                                          ║
║ ├─ PNG Images               : 2.6 MB (77.8%)            ║
║ ├─ Model Files (PKL)        : 608 KB (18.2%)            ║
║ ├─ Excel Files              : 15 KB (0.4%)              ║
║ ├─ Python Files             : 97 KB (2.9%)              ║
║ └─ Config Files             : 85 bytes (0%)             ║
╚══════════════════════════════════════════════════════════╝
```

---

## 🚀 Next Steps

### ✅ Sudah Selesai:
- [x] Clone repository
- [x] Copy folder PrediksiAPI
- [x] Configure git user
- [x] Add files to staging
- [x] Create commit
- [x] Push to GitHub

### 📌 Rekomendasi Selanjutnya:
- [ ] Buat README.md dengan dokumentasi project
- [ ] Buat .gitignore file
- [ ] Setup branch protection rules
- [ ] Enable GitHub Actions untuk CI/CD
- [ ] Create releases untuk versioning
- [ ] Add collaborators jika diperlukan

---

## 💡 Tips & Tricks

### 1. View Remote Status
```cmd
git remote -v
# Output:
# origin  https://github.com/fifiinvtsr/Skripsi-2025-2026.git (fetch)
# origin  https://github.com/fifiinvtsr/Skripsi-2025-2026.git (push)
```

### 2. View Commit Details
```cmd
git show a6cd28b
git log --oneline -10
```

### 3. Create Meaningful Commit Messages
```
Format: [TYPE] Subject (max 50 chars)

Types:
- feat: New feature
- fix: Bug fix
- docs: Documentation
- refactor: Code refactor
- test: Add tests
- update: Update existing code
```

---

## 🎓 Kesimpulan

Anda telah berhasil:
1. ✅ Setup Git di sistem lokal
2. ✅ Clone repository dari GitHub
3. ✅ Copy folder PrediksiAPI ke repository lokal
4. ✅ Add 30 files ke staging area
5. ✅ Create meaningful commit
6. ✅ Push ke GitHub cloud

**Total: 3.33 MB of project files uploaded! 🎉**

---

Generated: 10 Mei 2026 | Version: 1.0
