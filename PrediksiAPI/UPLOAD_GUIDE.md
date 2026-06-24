# Panduan Upload Folder PrediksiAPI ke GitHub

## Langkah-langkah Upload:

### 1. Persiapan - Install Git (jika belum)
Pastikan Git sudah terinstall di komputer Anda. Download dari: https://git-scm.com/

### 2. Clone Repository GitHub Anda
Buka Command Prompt (cmd) dan jalankan:

```cmd
cd C:\Users\LENOVO\Documents
git clone https://github.com/fifiinvtsr/Skripsi-2025-2026.git
cd Skripsi-2025-2026
```

### 3. Copy Folder PrediksiAPI ke Repository
```cmd
robocopy "C:\Users\LENOVO\Documents\PrediksiAPI" "C:\Users\LENOVO\Documents\Skripsi-2025-2026\PrediksiAPI" /E
```

Atau manual: Copy folder PrediksiAPI ke dalam folder Skripsi-2025-2026

### 4. Buka Folder Repository
```cmd
cd C:\Users\LENOVO\Documents\Skripsi-2025-2026
```

### 5. Check Status Git
```cmd
git status
```
Anda akan melihat file-file baru yang siap di-commit.

### 6. Add Semua File
```cmd
git add .
```
Atau add folder PrediksiAPI saja:
```cmd
git add PrediksiAPI/
```

### 7. Commit Changes
```cmd
git commit -m "Add PrediksiAPI folder with model and tests"
```

### 8. Push ke GitHub
```cmd
git push origin main
```

Jika diminta credential, masukkan:
- Username: fifiinvtsr
- Password: Gunakan Personal Access Token (PAT) jika sudah enable 2FA

### 9. Verifikasi di GitHub
Buka: https://github.com/fifiinvtsr/Skripsi-2025-2026
Anda akan melihat folder PrediksiAPI sudah terupload.

## Jika Belum Login GitHub di Git

### Setup First Time (jika belum setup)
```cmd
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

## Troubleshooting

### Jika ada error "permission denied"
- Pastikan folder tidak terbuka di aplikasi lain
- Gunakan Personal Access Token (PAT) sebagai password
- Create PAT di: https://github.com/settings/tokens

### Jika ada conflict
```cmd
git pull origin main
git push origin main
```

---
Selamat! Folder PrediksiAPI sudah terupload ke GitHub repository Anda.
