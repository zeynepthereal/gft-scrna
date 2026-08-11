# Veri Setleri

## Dataset 1 — CMV Kohort (İmmün Hücreler)

**Kaynak:** CZ CELLxGENE  
**Dosya:** `a3f06617-e433-4f73-8fed-24ff53408504.h5ad`  
**İndirme:** https://cellxgene.cziscience.com  
**Boyut:** ~320MB  

**İçerik:**
- N = 9.325 hücre
- k = 8 hücre tipi (CD4 T, CD8 T, NK, B cell, Monocyte, vb.)
- Hastalık: CMV (Sitomegalovirüs) enfeksiyonu + normal
- Batch düzeltme: Harmony PCA (X_pca_harmony)

**Kullanılan sütun:** `obs['predicted_AIFI_L2']`

---

## Dataset 2 — Meme Kanseri Tümör Mikro-Ortamı (TME)

**Kaynak:** CZ CELLxGENE — HTAN/HTAPP Broad  
**Dosya:** `ec51b9a3-ab71-4aba-86ff-1d04aae7f7cc.h5ad`  
**İndirme:** https://cellxgene.cziscience.com  
**Koleksiyon:** HTAN/HTAPP Broad — Spatio-molecular dissection of the breast cancer metastatic microenvironment  
**Boyut:** ~150MB  

**İçerik:**
- N = 12.493 hücre
- k = 9 hücre tipi (malignant, T cell, macrophage, fibroblast, endothelial, vb.)
- Doku: Karaciğer metastazı
- Hastalık: Meme kanseri (IDC)

**Kullanılan sütun:** `obs['cell_type']`

---

## İndirme Talimatları

```bash
# CZ CELLxGENE'den manuel indirme:
# 1. https://cellxgene.cziscience.com/datasets adresine git
# 2. Yukarıdaki dosya adlarını ara
# 3. .h5ad formatında indir
# 4. Bu klasöre koy

# Dosya adlarını kontrol et:
ls data/*.h5ad
```

## Etik ve Lisans

Her iki veri seti de kamuya açık, araştırma amaçlı kullanım için uygundur.
Orijinal yayınları atıflamayı unutmayın.
