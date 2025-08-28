import os # pentru separatorul de cale folosit la os.sep
from pathlib import Path # lipire cu /, teste cu .exists() etc
import argparse # pentru a citi argumente din linia de comanda ( --root, --meta etc)
import cv2 #pentru a citi si scrie imagini si pt a aplica filtrarea
import pandas as pd # citim meta.csv intr un data frame
from tqdm import tqdm # progress bar in terminal

IMG_EXT_CANDIDATES=[".tiff", ".tif", ".png", ".jpg", ".jpeg"] 

def load_meta(meta_path: Path)->pd.DataFrame: #citim fisierul meta
    meta_path=Path(meta_path) #convertim argumentul primit in Path
    if not meta_path.exists():
        raise FileNotFoundError(f"nu gasim fisierul meta: {meta_path}")
    if meta_path.suffix.lower() in [".csv"]: #alegem cititorul potrivit in functie de extensie
        df=pd.read_csv(meta_path)
    elif meta_path.suffix.lower() in [".xls", ".xlsx"]:
        df=pd.read_excel(meta_path)
    else:
        raise ValueError("format meta necunoscut (accept: .csv, .xls, .xlsx)")
   
    cols={c.lower(): c for c in df.columns}
    # 'id' si 'path' sunt obligatorii în dataset
    if "id" not in cols or "path" not in cols:
        raise KeyError("meta trebuie să contina coloanele 'id' și 'path'.")
    return df # returnam data frame ul


def find_image(root_dir: Path, subpath_str: str, base_id: str)->Path | None: # intoarce fie o cale valida, fie None
    """
    construim calea: <root>/<path>/<id>.<ext> incercand extensiile posibile
    """
    subpath=Path(str(subpath_str).strip().replace("\\", os.sep).replace("/", os.sep)) #taiem spatiile si inlocuim \ sau / cu separatorul pt sistem, os.sep si construim Path
    # merge deci si pe windows si pe linux

    for ext in IMG_EXT_CANDIDATES:
        cand=root_dir / subpath / f"{base_id}{ext}" #inceram toate extensiile si returnam daca exista fisierul
        if cand.exists():
            return cand
    return None

def mean_filter(src: Path, dst: Path, k: int = 5) -> bool:
    """
    aplicam mean/box filter (k x k). returneaza True dacă scrierea a reusit
    """
    img=cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False #citim sursa grayscale
    filt=cv2.blur(img, (k, k)) #aplicam box filter, cu media pe fereastra kxk, adica media pe un patrat de k x k in jurul fiecarui pixel
    dst.parent.mkdir(parents=True, exist_ok=True) #ne asiguram ca exista directorul destinatie
    return cv2.imwrite(str(dst), filt) #scriem imaginea filtrata

def preprocess_from_meta(root_dir: Path, meta_path: Path, out_root: Path, k: int=5, dry_run: bool=False):
    df=load_meta(meta_path) # citim meta ul
    root_dir=Path(root_dir) 
    out_root=Path(out_root) # convertim root si out in Path

    ok, miss, fail=0, 0, 0 # ok=cate fisiere am salvat, miss= nu am gasit fisierul meta, fail= am citit dar nu am reusit sa scriem iesirea

    it=tqdm(df.itertuples(index=False), total=len(df), desc="mean filtering (meta)") #transformam liniile din iterrows in itertuples pt ca sunt mai rapide
    #tqdm afiseaza progresul in terminal

    for row in it: #facem un map case insensitive

        row_dict=row._asdict() if hasattr(row, "_asdict") else row._asdict()
        keys={k.lower(): k for k in row_dict.keys()} # le facem cu litere mici
        id_val=str(row_dict[keys["id"]]).strip() # de exemplu, Ship_C06S01N0004 , scoatem spatiile etc cu strip
        path_val=str(row_dict[keys["path"]]).strip() # de exemplu, Cargo\BulkCarrier

        src=find_image(root_dir, path_val, id_val) #contruim calea catre fisier
        if src is None: # daca nu exista, o adaugam la miss
            miss+=1
            it.set_postfix(missed=miss, saved=ok, failed=fail) # actuamizam textul din progres bar si sarim la urm rand
            continue

        dst=out_root / Path(path_val) / src.name #facem calea de iesire, pastrand numele original, din datele primite (src.name)
        # de exemplu, data/preprocessed/Cargo/BulkCarrier/Ship_C06S01N0004.tiff

        if dry_run: #inseamna ca merge, chiar daca nu scriem nimic initial, e pentru verificare
            ok+=1
            it.set_postfix(missed=miss, saved=ok, failed=fail)
            continue #sarim la urm linie

        if mean_filter(src, dst, k=k): # daca nu e dry_run, chiar aplicam filtrul si scriem rezultatul
            ok+=1
        else:
            fail+=1
        it.set_postfix(missed=miss, saved=ok, failed=fail) 

    print(f"\nTerminare.\n  salvate: {ok}\n  lipsa fisier: {miss}\n  esec scriere: {fail}")

def build_argparser(): #definim argumentele CLI
    p=argparse.ArgumentParser(description="Mean filtering pe FUSAR-Ship folosind meta.csv/xls.")
    p.add_argument("--root", required=True, help="Directorul root al dataset-ului (ex: data/)")
    p.add_argument("--meta", required=True, help="Calea catre meta.csv sau meta.xls[x]")
    p.add_argument("--out",  required=True, help="Directorul de iesire (se pastreaza structura relativa)")
    p.add_argument("-k", "--kernel", type=int, default=5, help="Dimensiunea kernelului (impar recomandat)")
    p.add_argument("--dry-run", action="store_true", help="Nu scrie fisiere, doar verifica potrivirile")
    return p

if __name__ == "__main__":
    args=build_argparser().parse_args()
    preprocess_from_meta(
        root_dir=Path(args.root),
        meta_path=Path(args.meta),
        out_root=Path(args.out),
        k=args.kernel,
        dry_run=args.dry_run
    )

# python src\preprocess.py --root data --meta data\meta.csv --out data\preprocessed -k 5
# python src\preprocess.py --root data --meta data\meta.csv --out data\preprocessed --dry-run
