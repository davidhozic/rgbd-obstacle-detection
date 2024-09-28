# Segmentacija in detekcija ovir z RGB+D kamero
Seminarska naloga predmeta Robotski vid, na Fakulteti za elektrotehniko.

Minimalna zahtevana Python verzija: 3.9

David Hožič, Blaž Primšar.

[**Prenos demo posnetkov**](https://unilj-my.sharepoint.com/personal/dh8091_student_uni-lj_si/_layouts/15/download.aspx?SourceUrl=%2Fpersonal%2Fdh8091%5Fstudent%5Funi%2Dlj%5Fsi%2FDocuments%2Fposnetki%5Frv%2Fdata%2Ezip)

## Namestitev
- PyTorch CUDA:

    - (conda) ``conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia``
    - ali
    - (pip)   ``pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118``

- pip install -r requirements.txt

## Zagon
Demonstracijsko skripto (``demo.py``) se lahko zažene na dva načina. Prvi način je prek uporabe CLI vmesnika:

``python demo.py --help``

``python demo.py --task=detect_obstacles --input ./data/testni_podatki/testna_4 --output ./output/ovire/``

Drugi pa preko Python vmesnika (primer v ``main.py``):

```py
import demo

input_path = './data/testni_podatki/testna_4'
task = demo.ModelTask.DETECT_OBSTACLES  # / DETECT_DOORS / CALIB_ALIGN / CALIB_CAMERA
output_path = None # '<optional output folder path>'

# Vhod, način delovanja, (opcijski) izhod
demo.main(input_path=input_path, task=task, output_path=output_path)
```

## Konfiguracija parametrov
Demonstracijska skripta pravzaprav vsebuje le konfiguracijo, ki jo potrebuje sistem za pravilno delovanje, in zanko za branje slik.

Konfiguracija se nahaja na vrhu ``main()`` funkcije, kjer definirava:

- Kot nagnjenosti kamere čez horizontalno os ``angle_x`` - služi za transformacijo v sistem robota
- Višino kamere relativno na sistem robota ``camera_height`` - služi za transformacijo v sistem robota
- Afino matriko ``H_affine`` - služi za poravnavo globinske slike z glavno (barvno) sliko
- Homogeno tranformacijo preslikave iz sistema kamere v sistem robota ``H_cam_robot``
- Matriko kamere ``camera_matrix`` - služi transformaciji indeksov slikovne ravnine v
  prostorski koordinatni sistem kamere.
- (in druge vmesne stvari)

Nato definirava še glavni objekt tipa ``Camera``, čemur sledi zanka branja slik.

### !!! NUJNO !!!
Trenutna konfiguracija deluje dovolj dobro za postavitve kamere in intrinzične parametre na podatkovnih zbirkah
``testna_1``, ``testna_2`` in ``testna_4``. V zbirki ``testna_3`` pa intrinzični (matrika kamere) in
ekstrinzični (nagnjenost, višina, ipd.) preveč odstopajo od najinih, kar vodi v napačno filtriranje talnih objektov.

Priporočava, da za zbirko ``testna_3`` uporabite ``angle_x = np.deg2rad(-3)`` in ``camera_height = 345``.
Je že noter v kodi, le od-komentirati je treba. Ta dva parametra popravita filtriranje tal, še vedno pa se vidi
da meritve v kartezičnem prostoru niso ravno v redu, kar bo verjetno posledica napačnih intrinzičnih parametrov.


## Struktura projekta
Projekt je strukturiran v obliki manjšega Python paketa.
Paket se nahaja v isti mapi kot ta README in se imenuje ``camera``.
Znotraj paketa obstajajo štirje moduli:
- calibrator: vsebuje objekt ``Calibrator``, ki se uporablja za določanje intrinzičnih parametrov
  in za določitev afine transformacije, ki se uporablja za poravnavo globinske slike z barvno sliko.
- camera: glavni modul, ki vsebuje objekt ``Camera``. Tu notri se tudi izvaja vsa detekcija.
- data: modul podatkovnih struktur, ki predstavljajo rezultate detekcij.
- depthcamera: modul za procesiranje podatkov globinske kamere.

Poleg Python paketa, se na vrhu repozitorija nahajata ``models`` in ``scripts`` mapi.
Mapa ``models`` vsebuje FastSAM model ``FastSAM-s.pt`` in YOLO model (po meri treniran) ``vrata.pt``.
Mapa ``script`` vsebuje Python skripte za testiranje in skripto za pretvarjanje bag datotek v png slike.


## Načini delovanja
V demo (demo.py) skripti so na voljo štirje načini delovanja, definirani z ``demo.ModelTask`` ``IntEnum``-om:

- Detekcija ovir in estimacija kartezične pozicije - ``demo.ModelTask.DETECT_OBSTACLES``:

  V tem načinu se s FastSAM modelom segmentira in detektira objekte,
  jih filtrira, in oceni njihovo kartezično pozicijo.

- Detekcija dvižnih vrat oz. dvigala - ``demo.ModelTask.DETECT_DOORS``:

  V tem načinu se z YOLO modelom detektira prisotnost dvižnih vrat ali dvigala.

- Kalibracija afine poravnave - ``demo.ModelTask.CALIB_ALIGN``:

  Ta način odpre okno, kjer je potrebno izbrati korespondenčne točke na dveh slikah.
  Najprej se jih izbere na barvni sliki in nato v **istem vrstnem redu** še na normalizirani
  globinski sliki. Točke se postavlja s shift + enter, na naslednjo sliko pa se gre z dvojnim klikom.

- Kalibracija kamere (intrinzičnih parametrov) - ``demo.ModelTask.CALIB_CAMERA``:

  Omogoča kalibracijo na podlagi slik šahovnice.


Način delovanja se lahko izbere z:

- za CLI vmesnik: ``--task`` ``detect_obstacles`` / ``detect_doors`` / ``calib_align`` / ``calib_camera``.

  - Na primer, ``python demo.py --task detect_obstacles``.

- za Python vmesnik: ``task=demo.ModelTask.<napisano zgoraj>`` v klicu``main``
  funkcije.

