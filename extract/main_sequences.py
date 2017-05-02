import os

import numpy
import pandas
from sklearn.model_selection import train_test_split

from extract import c3d


def get_angle(a, b, c):
    # ritorna angolo centrato in a
    # calcolo vettori
    v1 = a - b
    v2 = a - c
    m = v1.dot((v2).T / (numpy.linalg.norm(v1, axis=1) * numpy.linalg.norm(v2, axis=1)))
    d = numpy.diag(m)
    return (numpy.arccos(numpy.clip(d, -1., 1.)))


# numpy.random.seed(7)

numpy.random.seed(8)
# cartella base
# qui mi aspetto dentro le dat_folders
BASE_FOLDER = "../data/"
# cartella con gli output
BASE_FOLDER_OUT = "../processed/"
TRAIN_FOLDER_OUT = "train/"
TEST_FOLDER_OUT = "test/"

# classi 0-3(faremo +1 al massimo)
# cartelle 0-3
# dentro le data folders ho i rispettivi contenuti degli zip(cartelle con id)
DATA_FOLDERS = ["0/", "1/", "2/", "3/"]

# classi associate alle data_folders
CLASSES = [i for i in range(len(DATA_FOLDERS))]
# parametri di interesse(presi da ferrari)
PARAMS = ["C7", "REP", "RUL", "RASIS", "RPSIS", "RCA", "RGT", "RLE",
          "RFM", "RA", "LEP", "LUL", "LASIS", "LPSIS", "LCA", "LGT", "LLE",
          "LFM", "LA"]
# angoli interessanti(a,b,c)
ANGLES_3D = [("LGT", "LPSIS", "LLE"), ("LLE", "LGT", "LCA"), ("LCA", "LLE", "LFM"), ("LEP", "LA", "LUL"),
             ("LEP", "C7", "LUL"),
             ("LLE", "LASIS", "LFM"), ("LA", "C7", "LEP"),
             ("RGT", "RPSIS", "RLE"), ("RLE", "RGT", "RCA"), ("RCA", "RLE", "RFM"), ("REP", "RA", "RUL"),
             ("REP", "C7", "RUL"),
             ("RLE", "RASIS", "RFM"), ("RA", "C7", "REP"),
             ("LPSIS", "LGT", "RGT"), ("LASIS", "LGT", "RGT"), ("LPSIS", "LLE", "RLE"), ("C7", "LA", "RA"),
             ("C7", "LEP", "REP"),
             ("RPSIS", "LGT", "RGT"), ("RASIS", "LGT", "RGT"), ("RPSIS", "LLE", "RLE"), ("C7", "LUL", "RUL"),
             ("LASIS", "C7", "LPSIS"), ("RASIS", "C7", "RPSIS"), ("LA", "LASIS", "RASIS"), ("RA", "LASIS", "RASIS")]



NUM_PARAMS = len(PARAMS)
# PARAMETRI CONTROLLO
TRAIN_SIZE = 0.75
# prendiamo angoli 3D proiettati
CONF = ["angles3D"]
#inverto assi se ha camminato sul lato corto
SWAP_AXIS = True
ALLOW_DIS = True
# frame vicon
FRAME_VICOM = 100
# frame che vogliamo ogni secondo
FRAME_PER_SECOND = 50
# guardiamo che sia divisibile
assert FRAME_VICOM % FRAME_PER_SECOND == 0
# campiono ogni FRAME_VICOM/FRAME_PER_SECOND
FRAME_COUNT = FRAME_VICOM / FRAME_PER_SECOND
# NUMERO SEQUENZE MASSIME PER FILE
SEQ_NUM = 45
# LUNGHEZZA SEQUENZE
SEQ_LEN = 75
# DISPLACE TRA SEQUENZE CONSECUTIVE
DISPLACE = 15
# labels classi
labels_classes_train = []
labels_classes_test = []
# labels pazienti
labels_patients_train = []
labels_patients_test = []
# not valid
invalid = 0
# controllo asse principale
top_axes_dist = {0: 0, 1: 0, 2: 0}
# lista classi che salveremo in un file a parte
# lista perche non ho voglia di contare tutti i file,e piu lenta ma tanto lo lanciamo solo una volta
j = 0
len_step = []
for i in range(len(CLASSES)):
    # entro nella cartella classe
    folder_class = BASE_FOLDER + DATA_FOLDERS[i]
    print "start working in folder {}".format(folder_class)
    # qui divido le cartelle
    train_labels, test_labels = train_test_split(sorted(os.listdir(folder_class)), train_size=TRAIN_SIZE)

    # per ogni cartella paziente estraggo i c3d
    for folder_patient in sorted(os.listdir(folder_class)):
        label_patient = folder_patient
        folder_patient = folder_class + folder_patient
        # lista file
        for file_name in [file_name for file_name in sorted(os.listdir(folder_patient)) if file_name[-3:] == "c3d"]:
            # reader per leggere il c3d
            file_name = folder_patient + "/" + file_name
            try:
                reader = c3d.Reader(open(file_name, 'rb'))
            except Exception:
                print "eccezione in {}".format(file_name)
                continue
            first_frame = reader.header.first_frame
            last_frame = reader.header.last_frame
            #mi servono gli eventi di footstrike
            context_header = [s.strip() for s in reader.groups['EVENT'].params['CONTEXTS'].string_array]
            label_header = [s.strip() for s in reader.groups['EVENT'].params['LABELS'].string_array]
            times_header = reader.groups['EVENT'].params['TIMES'].float_array
            times_header = times_header.flatten()
            times_header = times_header[times_header != 0]
            df = pandas.DataFrame(data={"context": context_header, "label": label_header, "times": times_header})
            df = df.sort_values(by="times")
            df.times = (df.times * 100).astype("int32")
            # scartiamo se non ci sono abbastanza eventi interessanti
            if len(df.times) < 9:
                continue
            # prendiamo dal occ
            occ = 0
            frame = min(df[(df.context == "Right") & (df.label == "Foot Strike")].iloc[occ].times,
                        df[(df.context == "Left") & (df.label == "Foot Strike")].iloc[occ].times)

            # scartiamo se il frame e negativo
            if frame - first_frame < 0:
                continue
            # estraggo labels dei points
            params = [param.strip() for param in reader.get("POINT").get("LABELS").string_array]
            # guardo che ci siano tutti quelli del file matlab di Ferrari
            # sono 52 quelli che non l hanno
            # visto che ogni paziente ha piu prove non dovrebbe essere un problema
            if len([param for param in PARAMS if param not in params]) > 0:
                continue
            # ci sono tutti ma non so in che ordine!
            # mi servono gli indici dei param in params
            params = [params.index(param) for param in PARAMS]

            # li usero per accedere ai frame
            # ogni frame ha NUM_PARAMS*5 coordinate(xyz + precisione e altro)
            # devo salvare le sequenze
            # salvo un elemento in piu che usero come label
            # ho numero_sequenze*lunghezza_sequenza+1*numero_marker*3

            index_camp = FRAME_COUNT
            index = 0
            # punti del file

            points = numpy.zeros((last_frame - first_frame + 1, 3, len(PARAMS), 3))
            # punti 3 piani
            points_1 = numpy.zeros((last_frame - first_frame + 1, len(PARAMS), 3))
            points_2 = numpy.zeros((last_frame - first_frame + 1, len(PARAMS), 3))
            points_3 = numpy.zeros((last_frame - first_frame + 1, len(PARAMS), 3))

            for k, point, analog in reader.read_frames():
                if k >= frame:
                    # se allow permetto di iniziare a campionare anche dopo frame
                    # ma non permetto comunque di saltare fotogrammi
                    # infatti funziona solo finche index e 0
                    if ALLOW_DIS and -1 in point[params, 3:] and index == 0:
                        continue
                    # prendo un campione solo se index_camp = FRAME_COUNT
                    if index_camp == FRAME_COUNT:
                        # campiono

                        # se invalido esco
                        if -1 in point[params, 3:]:
                            break

                        point_fix = point[params, 0:3]
                        # calcolo primo piano
                        # piano meta fronte in avanti
                        c7 = point_fix[PARAMS.index("C7")]
                        lgt = point_fix[PARAMS.index("LGT")]
                        rgt = point_fix[PARAMS.index("RGT")]
                        lasis = point_fix[PARAMS.index("LASIS")]
                        rasis = point_fix[PARAMS.index("RASIS")]
                        lpsis = point_fix[PARAMS.index("LPSIS")]
                        rpsis = point_fix[PARAMS.index("RPSIS")]

                        coef = lgt - rgt
                        delta = -(coef.dot(c7))
                        points_1[index] = point_fix - (numpy.sum(point_fix * coef, axis=1) + delta).reshape(-1,
                                                                                                            1) * coef / numpy.linalg.norm(
                            coef) ** 2

                        # calcolo primo piano
                        # piano spalle culo
                        coef_1 = c7 - lgt
                        coef_2 = c7 - rgt
                        coef = numpy.cross(coef_1, coef_2)
                        delta = -(coef.dot(c7))
                        points_2[index] = point_fix - (numpy.sum(point_fix * coef, axis=1) + delta).reshape(-1,
                                                                                                            1) * coef / numpy.linalg.norm(
                            coef) ** 2

                        # calcolo terzo piano
                        # pavimento
                        coef = numpy.array((0, 0, 1))
                        delta = 0.0
                        points_3[index] = point_fix - (numpy.sum(point_fix * coef, axis=1) + delta).reshape(-1,
                                                                                                            1) * coef / numpy.linalg.norm(
                            coef) ** 2

                        # incremento indice
                        index += 1
                        # camp da 1!!!
                        index_camp = 1
                    else:
                        index_camp += 1

            # prendo quelli validi
            points_1 = points_1[0:index]
            points_2 = points_2[0:index]
            points_3 = points_3[0:index]
            #genero un solo vettore
            points = numpy.concatenate((points_1, points_2, points_3), axis=1)

            # controllo che ci sia qualcosa
            if index <= SEQ_LEN:
                print "non ci sono abbastanza dati per una sequenza in {}!".format(file_name)
                invalid += 1
                continue

            # controllo ordine assi
            if SWAP_AXIS:
                # questo lavora sempre sui relativi
                # ma va confrontato con i risultati di Gabri
                # sommo i marker ma in valore assoluto
                top = numpy.argmax(numpy.sum(numpy.abs(numpy.diff(points, axis=0)).reshape(-1, 3), axis=0))
                # aggiorno contatori
                top_axes_dist[top] += 1
                # uno e il piu frequente, ma molte volte e zero
                # se e zero lo metto in posizione uno
                # due non c'e mai
                if top == 0:
                    points = points[:, :, [1, 0, 2]]

            # la len e il numero di punti
            arr = []
            for conf in CONF:
                if conf == "angles3D":
                    angles = numpy.zeros((len(points), 3 * len(ANGLES_3D)))

                    for k, angle in enumerate(ANGLES_3D):
                        index_angle_a = PARAMS.index(angle[0])
                        index_angle_b = PARAMS.index(angle[1])
                        index_angle_c = PARAMS.index(angle[2])

                        angles[:, k] = get_angle(points[:, index_angle_a], points[:, index_angle_b],
                                                 points[:, index_angle_c])

                        angles[:, k + len(ANGLES_3D)] = get_angle(points[:, index_angle_a + len(PARAMS)],
                                                                  points[:, index_angle_b + len(PARAMS)],
                                                                  points[:, index_angle_c + len(PARAMS)])
                        angles[:, k + len(ANGLES_3D) * 2] = get_angle(points[:, index_angle_a + len(PARAMS) * 2],
                                                                      points[:, index_angle_b + len(PARAMS) * 2],
                                                                      points[:, index_angle_c + len(PARAMS) * 2])

                    temp = angles


                else:
                    raise NotImplementedError
                arr.append(temp[..., numpy.newaxis])
            # posso concatenare tutto
            # ma devo considerare le lunghezze
            #cioe se ho piu configurazioni(e stato lasciato per sicurezza)
            min_len = min([len(el) for el in arr])
            points = numpy.concatenate([el[-min_len:] for el in arr], axis=1)
            # sequenze
            seq_numpy = numpy.zeros((SEQ_NUM, SEQ_LEN, points.shape[1], points.shape[2]))
            # contatori
            #itereativamente provo a prendere sequenze da points
            seq_index = 0
            base_index = 0
            while True:
                # provo a prendere una sequenza
                seq = points[base_index:base_index + SEQ_LEN]
                if len(seq) != SEQ_LEN:
                    break
                # sono riuscito ad avere una nuova sequenza
                seq_numpy[seq_index] = seq
                lab = points[base_index + SEQ_LEN:base_index + SEQ_LEN + 1]
                if len(lab) != 1:
                    break
                # avanzo di DISPLACE
                base_index += DISPLACE
                # avanzo seq_index
                seq_index += 1
                if seq_index == SEQ_NUM:
                    break
            # prendo solo quelle ottenute
            seq_numpy = seq_numpy[0:seq_index]

            # potrei essere qui ma non avere  dei dati
            if seq_index == 0:
                # non ho neanche una sequenza!
                print "non ci sono abbastanza dati per una sequenza in {}!".format(file_name)
                invalid += 1
                continue
            else:
                print "trovate {} sequenze in {}! ".format(seq_index, file_name)

                # salvo in base a se train o test
                if label_patient in train_labels:
                    numpy.savez(BASE_FOLDER_OUT + TRAIN_FOLDER_OUT + "{}.c3d".format(j), seq_numpy)
                    # labels class
                    labels_classes_train.extend([i] * seq_index)
                    # labels patient
                    labels_patients_train.extend([int(label_patient)] * seq_index)
                else:
                    numpy.savez(BASE_FOLDER_OUT + TEST_FOLDER_OUT + "{}.c3d".format(j), seq_numpy)
                    # labels class
                    labels_classes_test.extend([i] * seq_index)
                    # labels patient
                    labels_patients_test.extend([int(label_patient)] * seq_index)
                j += 1

    print "end working in folder {}".format(folder_class)
    print "-------"
# train
numpy.save(BASE_FOLDER_OUT + TRAIN_FOLDER_OUT + "labels_classes", numpy.array(labels_classes_train))
numpy.save(BASE_FOLDER_OUT + TRAIN_FOLDER_OUT + "labels_patients", numpy.array(labels_patients_train))
# test
numpy.save(BASE_FOLDER_OUT + TEST_FOLDER_OUT + "labels_classes", numpy.array(labels_classes_test))
numpy.save(BASE_FOLDER_OUT + TEST_FOLDER_OUT + "labels_patients", numpy.array(labels_patients_test))
print "invalid {}".format(invalid)
print "axis dist {}".format(top_axes_dist)
print "end"
