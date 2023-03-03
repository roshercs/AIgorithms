from django.shortcuts import render
from django.http import HttpResponse,JsonResponse
from django.shortcuts import get_object_or_404,render,redirect
from .forms import ParamsApriori
import pandas as pd                 # Para la manipulación y análisis de los datos
import numpy as np                  # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt     # Para la generación de gráficas a partir de los datos
from apyori import apriori
from django.core.files.storage import FileSystemStorage
import dataframe_image as dfi
from os import remove
import glob
from scipy.spatial.distance import cdist    # Para el cálculo de distancias
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import seaborn as sns             # Para la visualización de datos basado en matplotlib
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from kneed import KneeLocator
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
from mpl_toolkits.mplot3d import Axes3D
from sklearn import model_selection
from sklearn import linear_model
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import RocCurveDisplay
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier
global filename
global uploaded_file_url
global DatosTransacciones
global lift
global support
global confidence
global ListaM
global tamList
global listRel
global ListaData
global metrica
global datosProc
global lam
global decimales
global MProcesada
global MetricasDistancia
global clustering_type
global column_name
global colNames
global DatosSelect
global kvalues
global cluster_table
global MatrizDatos
global DatosDroped
global ConteoCluster
global CentroidesH
global varClas
global size_train
global varDep
global MatrizDatosAux
global valoresClas
global X_train
global X_validation
global Y_train
global Y_validation
global X_train_size
global Y_train_size
global exactitud
global Probabilidad
global Matriz_Clasificacion
global intercepto
global coeficientes
global estadoResultado
global auxiliar
global listado
global ValoresPredic
global ClasificacionRL
global maxDepth
global minSampleLeaf
global minSampleSplit
global typeTree
global typeClass
global ClasificacionAD
global ValoresMod1
global ImportanciaMod1
global nestimators
# Create your views here.
def index(request):
    return render(request,'index.html')
def apriori_page(request):
    py_files=glob.glob('./media/*')
    for py_file in py_files:
        try:
            remove(py_file)
        except OSError as e:
            print(f"Error:{ e.strerror}")
    return render(request,'apriori.html')

def base(request):
    return render(request,'layouts/base.html')
def prueba(request):
    return render(request,'prueba.html')


def GetValuesApriori(request):
    if request.method=='POST':
        global lift
        global support
        global confidence
        lift=request.POST['lift']
        support=request.POST['support']
        confidence=request.POST['confidence']
        myfile=request.FILES['file_name']
        fs=FileSystemStorage()
        global filename
        filename=fs.save(myfile.name,myfile)
        global uploaded_file_url
        uploaded_file_url=fs.url(filename)
        return redirect('/apriori_result.html/')
    else:
        py_files=glob.glob('./media/*')
        for py_file in py_files:
            try:
                remove(py_file)
            except OSError as e:
                print(f"Error:{ e.strerror}")
        return render(request,'apriori_simulate.html')
def AprioriResults(request):
    global filename
    global lift
    global support
    global confidence
    global tamList
    AprioriAlgorithm()
    saludo="Estos son los datos ingresados: "
    return render(request,'apriori_result.html',{
        'filename': filename,
        'hola': saludo,
        'lift': lift,
        'confidence': confidence,
        'support': support,
        'tamano': tamList,
        'listado': listRel,
        'data': Datos,
    })

def AprioriAlgorithm():
    global Datos
    global ListaM
    global uploaded_file_url
    global lift
    global support
    global confidence
    global tamList
    global listRel
    global ListaData
    Datos = pd.read_csv(uploaded_file_url,header=None)
    Transacciones = Datos.values.reshape(-1).tolist()
    ListaM = pd.DataFrame(Transacciones)
    ListaM=pd.DataFrame(Transacciones)
    ListaM['Frecuencia'] = 1
    ListaM = ListaM.groupby(by=[0], as_index=False).count().sort_values(by=['Frecuencia'], ascending=True)
    ListaM['Porcentaje'] = (ListaM['Frecuencia'] / ListaM['Frecuencia'].sum())
    ListaM = ListaM.rename(columns={0 : 'Item'})
    plt.figure(figsize=(16,20), dpi=300)
    plt.ylabel('Item')
    plt.xlabel('Frecuencia')
    plt.barh(ListaM['Item'], width=ListaM['Frecuencia'], color='blue')
    plt.savefig("./media/DistribucionApriori.jpg", bbox_inches='tight')
    ListaTrans = Datos.stack().groupby(level=0).apply(list).tolist()
    listRel=[{}]
    Reglas= apriori(ListaTrans, min_support=float(support),min_confidence=float(confidence),min_lift=float(lift))
    j=1
    for item in Reglas:
        #El primer índice de la lista
        Emparejar = item[0]
        items = [x for x in Emparejar]
        listRel.append({'index':j,'item_base':str(item[2][0][0]),'consecuente':str(item[2][0][1]),'soporte':str(item[1]),'confianza': str(item[2][0][2]),'elevacion':str(item[2][0][3])})
        j+=1
    listRel.remove({})
    tamList=len(listRel)
    imprimeMensaje()
    print(listRel)

def Metricas(request):
    return render(request,'metricas.html')
def Metricas_simulate(request):
    return render(request,'metricas_simulate.html')
def Metricas_result(request):
    return render(request,'metricas_result.html')

def GetValuesMetricas(request):
    if request.method=='POST':
        global metrica
        global datosProc
        global lam
        global decimales
        metrica=request.POST['medicion']
        datosProc=request.POST['datosProc']
        lam=request.POST['lambda']
        decimales=request.POST['decimales']
        myfile=request.FILES['file_name']
        fs=FileSystemStorage()
        global filename
        filename=fs.save(myfile.name,myfile)
        global uploaded_file_url
        uploaded_file_url=fs.url(filename)
        return redirect('/metricas_result.html/')
    else:
        py_files=glob.glob('./media/*')
        for py_file in py_files:
            try:
                remove(py_file)
            except OSError as e:
                print(f"Error:{ e.strerror}")
        return render(request,'metricas_simulate.html')
def MetricasResults(request):
    global filename
    global metrica
    global datosProc
    global lam
    global decimales
    global Datos
    global MProcesada
    global MetricasDistancia
    MetricasAlgorithm()
    saludo="Estos son los datos ingresados: "
    return render(request,'metricas_result.html',{
        'filename': filename,
        'hola': saludo,
        'data': MProcesada,
        'decimales': decimales,
        'lambda': lam,
        'datosProc':datosProc,
        'metrica':metrica,
        'resultado':MetricasDistancia,
    })

def MetricasAlgorithm():
    global Datos
    global ListaM
    global uploaded_file_url
    global metrica
    global datosProc
    global lam
    global decimales
    global filename
    global MProcesada
    global MetricasDistancia
    Datos = pd.read_csv('media/'+filename)
    estandarizar = StandardScaler() 
    normalizar = MinMaxScaler()
    lb=float(lam)
    if datosProc=='estandarizados':
        MProcesada=estandarizar.fit_transform(Datos)
        MProcesada=pd.DataFrame(MProcesada)
    elif datosProc=='normalizados':
        MProcesada=normalizar.fit_transform(Datos)
        MProcesada=pd.DataFrame(MProcesada)
    else:
        MProcesada=Datos
    
    if metrica=="minkowski":
        Dst = cdist(MProcesada, MProcesada, metric=metrica,p=lb)
    else:
        Dst = cdist(MProcesada, MProcesada, metric=metrica)
    MetricasDistancia=pd.DataFrame(Dst)
    MetricasDistancia=MetricasDistancia.round(int(decimales))

def Seleccion(request):
    return render(request,'seleccion_caracteristicas.html')
def Clustering(request):
    return render(request,'clustering.html')

def GetValuesClustering(request):
    if request.method=='POST':
        global metrica
        global datosProc
        global clustering_type
        global Datos
        global column_name
        metrica=request.POST['medicion']
        datosProc=request.POST['datosProc']
        clustering_type=request.POST['clustering']
        myfile=request.FILES['file_name']
        fs=FileSystemStorage()
        global filename
        filename=fs.save(myfile.name,myfile)
        global uploaded_file_url
        uploaded_file_url=fs.url(filename)
        Datos = pd.read_csv('media/'+filename)
        column_name=[{}]
        for x in Datos:
            column_name.append(x)
        column_name.remove({})
        CorrDatos = Datos.corr(method='pearson')
        plt.figure(figsize=(14,7))
        MatrizInf = np.triu(CorrDatos)
        sns.heatmap(CorrDatos, cmap='RdBu_r', annot=True, mask=MatrizInf)
        plt.savefig("./media/Correlacion.jpg", bbox_inches='tight')
        return redirect('/clustering_simulate_2.html/')
    else:
        py_files=glob.glob('./media/*')
        for py_file in py_files:
            try:
                remove(py_file)
            except OSError as e:
                print(f"Error:{ e.strerror}")
        return render(request,'clustering_simulate.html')
def GetValuesClustering2(request):
    global column_name
    global clustering_type
    global metrica
    global datosProc
    global lam
    global colNames
    global Datos
    global MProcesada
    global DatosSelect
    global MatrizDatos
    global DatosDroped
    global kvalues
    print("Columnas: ")
    print(column_name)
    if request.method=='POST':
        aux=request.POST.getlist('columns')
        for j in aux:
            auxb=j[:-1]
            column_name.remove(auxb)
            Datos=Datos.drop([auxb],axis=1)
        print(column_name)
        MatrizDatos=np.array(Datos[column_name])
        print(Datos)
        estandarizar=StandardScaler()
        normalizar=MinMaxScaler()
        if datosProc=="estandarizados":
            MProcesada=estandarizar.fit_transform(MatrizDatos)
        else:
            MProcesada=normalizar.fit_transform(MatrizDatos)
        if clustering_type=="particional":
            return redirect('/clustering_simulate_3.html/')
        return redirect('/clustering_result.html/')
    else:
        return render(request,'clustering_simulate_2.html',{
            'columnas':column_name,
            'clustering':clustering_type,
        })

def GetValuesClustering3(request):
    global colNames
    global column_name
    global MProcesada
    global kvalues
    if request.method=='POST': 
        kvalues=request.POST['kvalue']
        return redirect('/clustering_result.html/')
    else:
        SSE = []
        for i in range(2, 12):
            km = KMeans(n_clusters=i, random_state=0) #el estado inicial es 0
            km.fit(MProcesada)
            SSE.append(km.inertia_)
        plt.figure(figsize=(10, 7))
        plt.plot(range(2, 12), SSE, marker='o')
        plt.xlabel('Cantidad de clusters *k*')
        plt.ylabel('SSE')
        plt.title('Elbow Method')
        plt.savefig("./media/elbow.jpg", bbox_inches='tight')
        kl = KneeLocator(range(2, 12), SSE, curve="convex", direction="decreasing")
        kvalues=kl.elbow
        return render(request,'clustering_simulate_3.html',{
            'k':kvalues
        })
def ClusteringResults(request):
    global filename
    global metrica
    global datosProc
    global lam
    global clustering_type
    global cluster_table
    global DatosDroped
    global Datos
    global ConteoCluster
    global CentroidesH
    global kvalues
    ClusteringAlgorithm()
    if clustering_type=="particional":
        return render(request,'clustering_result.html',{
            'filename':filename,
            'datosProc':datosProc,
            'clusteringtype':clustering_type,
            'data': Datos,
            'conteo': ConteoCluster,
            'centroides': CentroidesH,
            'kvalues':kvalues,
            'metrica':metrica
        })
    else:
        return render(request,'clustering_result.html',{
            'filename':filename,
            'datosProc':datosProc,
            'clusteringtype':clustering_type,
            'data': Datos,
            'conteo': ConteoCluster,
            'centroides': CentroidesH,
            'kvalues':kvalues,
            'metrica':metrica
        })

    

def ClusteringAlgorithm():
    global MProcesada
    global clustering_type
    global metrica
    global kvalues
    global Datos
    global DatosDroped
    global ConteoCluster
    global column_name
    global CentroidesH
    if clustering_type=="particional":
        MParticional = KMeans(n_clusters=int(kvalues), random_state=0).fit(MProcesada)
        MParticional.predict(MProcesada)
        MParticional.labels_
        Datos['clusterP']= MParticional.labels_
        aux=Datos.groupby(['clusterP'])['clusterP'].count()
        ConteoCluster=pd.DataFrame(aux)
        CentroidesH = Datos.groupby(['clusterP'])[column_name].mean()
        i=len(CentroidesH)
        plt.rcParams['figure.figsize'] = (10, 7)
        plt.style.use('ggplot')
        colores=['red', 'blue', 'green', 'yellow', 'cian','black','dimgray','skyblue','lightgreen','magenta','brown','violet']
        asignar=[]
        for row in MParticional.labels_:
            asignar.append(colores[row])
        
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(MProcesada[:, 0], 
                MProcesada[:, 1], 
                MProcesada[:, 2], marker='o', c=asignar, s=60)
        ax.scatter(MParticional.cluster_centers_[:, 0], 
                MParticional.cluster_centers_[:, 1], 
                MParticional.cluster_centers_[:, 2], marker='o', c=colores[0:i], s=1000)
        plt.savefig("./media/clusters_3d.jpg", bbox_inches='tight')
    else:
        plt.figure(figsize=(15, 10))
        plt.title("Clusters de datos")
        plt.xlabel('Observaciones')
        plt.ylabel('Distancia')
        Arbol = shc.dendrogram(shc.linkage(MProcesada, method='complete', metric=metrica))
        auxiliar=Arbol['leaves_color_list']
        auxb=np.unique(auxiliar)
        kvalues=len(auxb)
        plt.savefig("./media/ClusteringJerarquico.jpg", bbox_inches='tight')
        MJerarquico = AgglomerativeClustering(n_clusters=kvalues, linkage='complete', affinity=metrica)
        MJerarquico.fit_predict(MProcesada)
        Datos['clusterH'] = MJerarquico.labels_
        aux=Datos.groupby(['clusterH'])['clusterH'].count() 
        ConteoCluster=pd.DataFrame(aux)
        CentroidesH = Datos.groupby(['clusterH'])[column_name].mean()
        plt.figure(figsize=(10, 7))
        plt.scatter(MProcesada[:,0], MProcesada[:,1], c=MJerarquico.labels_)
        plt.grid()
        plt.savefig("./media/clusters_graph.jpg", bbox_inches='tight')




def RegresionLogistica(request):
    return render(request,'regresion_logistica.html')

def GetValuesRegretion(request):
    global column_name
    if request.method=='POST':
        global Datos
        global varClas
        global size_train
        size_train=request.POST['train']
        myfile=request.FILES['file_name']
        fs=FileSystemStorage()
        global filename
        filename=fs.save(myfile.name,myfile)
        global uploaded_file_url
        uploaded_file_url=fs.url(filename)
        Datos = pd.read_csv('media/'+filename)
        column_name=[{}]
        for x in Datos:
            column_name.append(x)
        column_name.remove({})
        return redirect('/regresion_simulate_2.html/')
    else:
        py_files=glob.glob('./media/*')
        for py_file in py_files:
            try:
                remove(py_file)
            except OSError as e:
                print(f"Error:{ e.strerror}")
        return render(request,'regresion_simulate.html')

def GetValuesRegretion2(request):
    global colNames
    global column_name
    global Datos
    global MProcesada
    global DatosSelect
    global MatrizDatos
    global MatrizDatosAux
    global DatosDroped
    global varDep

    if request.method=='POST':
        varDep=request.POST['varOP']
        column_name.remove(varDep)
        DatosAux=Datos.drop([varDep],axis=1)
        CorrDatos = DatosAux.corr(method='pearson')
        plt.figure(figsize=(14,7))
        MatrizInf = np.triu(CorrDatos)
        sns.heatmap(CorrDatos, cmap='RdBu_r', annot=True, mask=MatrizInf)
        plt.savefig("./media/Correlacion.jpg", bbox_inches='tight')
        return redirect('/regresion_simulate_3.html')
    else:
        return render(request,'regresion_simulate_2.html',{
            'columnas':column_name
        })
def GetValuesRegretion3(request):
    global column_name
    global colNames
    global Datos
    global MProcesada
    global DatosSelect
    global MatrizDatos
    global DatosDroped

    if request.method=='POST':
        aux=request.POST.getlist('columns')
        DatosDroped=Datos
        for j in aux:
            auxb=j[:-1]
            column_name.remove(auxb)
            DatosDroped=DatosDroped.drop([auxb],axis=1)
        MatrizDatos=np.array(Datos[column_name])
        print(column_name)
        RegresionAlgorithm()
        return redirect('/regresion_result.html/')
    else:
        print("PRUEBA DE ALCANCE")
        print(column_name)
        return render(request,'regresion_simulate_3.html',{
            'columnas':column_name,
        })

def RegresionResults(request):
    global filename
    global Datos
    global estadoResultado
    global auxiliar
    global X_train_size
    global X_train
    global Y_train_size
    global exactitud
    global Probabilidad
    global Matriz_Clasificacion
    global intercepto
    global coeficientes
    global column_name
    global listado
    global ValoresPredic
    global varDep
    valores1=ValoresPredic[0]
    valores2=ValoresPredic[1]
    X_train_frame=pd.DataFrame(X_train)
    if request.method=='POST':
        AuxPred={}
        for x in column_name:
            AuxPred[x]=[float(request.POST[x])]
        Sujeto = pd.DataFrame(AuxPred)
        clasificado=ClasificacionRL.predict(Sujeto)
        estadoResultado="1"
        return render(request,'regresion_result.html',{
            'filename':filename,
            'data': Datos,
            'estado':estadoResultado,
            'Xsize':X_train_size,
            'Ysize':Y_train_size,
            'exact':exactitud,
            'prob':Probabilidad,
            'MatClas':Matriz_Clasificacion,
            'inter':intercepto,
            'coef':coeficientes,
            'lista':column_name,
            'valores1':valores1,
            'valores2':valores2,
            'clase': varDep,
            'entrenamiento':X_train_frame,
            'clas':clasificado,
        })
    else:
        estadoResultado="0"
        clasificado=0
        return render(request,'regresion_result.html',{
            'filename':filename,
            'data': Datos,
            'estado':estadoResultado,
            'Xsize':X_train_size,
            'Ysize':Y_train_size,
            'exact':exactitud,
            'prob':Probabilidad,
            'MatClas':Matriz_Clasificacion,
            'inter':intercepto,
            'coef':coeficientes,
            'lista':column_name,
            'valores1':valores1,
            'valores2':valores2,
            'clase': varDep,
            'entrenamiento':X_train_frame,
            'clas':clasificado,
        })

    

def RegresionAlgorithm():
    global Datos
    global column_name
    global valoresClas
    global varDep
    global DatosDroped
    global X
    global Y
    global X_train
    global X_validation
    global Y_train
    global X_train_size
    global Y_train_size
    global Y_validation
    global size_train
    global exactitud
    global Probabilidad
    global Matriz_Clasificacion
    global intercepto
    global coeficientes
    global ValoresPredic
    global ClasificacionRL
    ValoresPredic=Datos[varDep].unique().tolist()
    Datos=Datos.replace({ValoresPredic[0]: 0, ValoresPredic[1]: 1})
    X = np.array(DatosDroped[column_name])
    Y = np.array(Datos[[varDep]])
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size = float(size_train), 
                                                                random_state = 0,
                                                                shuffle = True)
    X_train_size=len(X_train)
    Y_train_size=len(Y_train)
    ClasificacionRL = linear_model.LogisticRegression()
    ClasificacionRL.fit(X_train, Y_train)
    ClasificacionRL = linear_model.LogisticRegression()
    ClasificacionRL.fit(X_train, Y_train)
    Probabilidad = ClasificacionRL.predict_proba(X_validation)
    Probabilidad=Probabilidad.round(4)
    Probabilidad=pd.DataFrame(Probabilidad)
    Y_ClasificacionRL = ClasificacionRL.predict(X_validation)
    exactitud=accuracy_score(Y_validation, Y_ClasificacionRL)
    ModeloClasificacion = ClasificacionRL.predict(X_validation)
    Matriz_Clasificacion = pd.crosstab(Y_validation.ravel(), 
                                    ModeloClasificacion, 
                                    rownames=['Reales'], 
                                    colnames=['Clasificación']) 
    CurvaROC = RocCurveDisplay.from_estimator(ClasificacionRL, X_validation, Y_validation, name=varDep)
    plt.savefig("./media/ROC.jpg", bbox_inches='tight')
    intercepto="Intercept:", (ClasificacionRL.intercept_)
    coeficientes='Coeficientes:', (ClasificacionRL.coef_)
    
    
    

def Arboles(request):
    return render(request,'arboles_aleatorios.html')

def GetValuesArboles(request):
    global column_name
    if request.method=='POST':
        global Datos
        global varClas
        global size_train
        global maxDepth
        global minSampleLeaf
        global minSampleSplit
        global typeTree
        global typeClass
        size_train=request.POST['train']
        maxDepth=request.POST['depth']
        minSampleLeaf=request.POST['minSamples']
        minSampleSplit=request.POST['minSplit']
        typeTree=request.POST['tipo']
        typeClass=request.POST['clase']
        myfile=request.FILES['file_name']
        fs=FileSystemStorage()
        global filename
        filename=fs.save(myfile.name,myfile)
        global uploaded_file_url
        uploaded_file_url=fs.url(filename)
        Datos = pd.read_csv('media/'+filename)
        column_name=[{}]
        for x in Datos:
            column_name.append(x)
        column_name.remove({})
        return redirect('/arboles_simulate_2.html/')
    else:
        py_files=glob.glob('./media/*')
        for py_file in py_files:
            try:
                remove(py_file)
            except OSError as e:
                print(f"Error:{ e.strerror}")
        return render(request,'arboles_simulate.html')

def GetValuesArboles2(request):
    global column_name
    global Datos
    global varDep
    global typeTree
    global nestimators
    if request.method=='POST':
        if typeTree=="forest":
            nestimators=request.POST['nestimators']
        else:
            nestimators=1
        varDep=request.POST['varOP']
        column_name.remove(varDep)
        DatosAux=Datos.drop([varDep],axis=1)
        CorrDatos = DatosAux.corr(method='pearson')
        plt.figure(figsize=(14,7))
        MatrizInf = np.triu(CorrDatos)
        sns.heatmap(CorrDatos, cmap='RdBu_r', annot=True, mask=MatrizInf)
        plt.savefig("./media/Correlacion.jpg", bbox_inches='tight')
        return redirect('/arboles_simulate_3.html')
    else:
        return render(request,'arboles_simulate_2.html',{
            'columnas':column_name,
            'tipo':typeTree,
        })
def GetValuesArboles3(request):
    global column_name
    global Datos
    global MatrizDatos
    global DatosDroped
    global typeTree
    if request.method=='POST':
        aux=request.POST.getlist('columns')
        DatosDroped=Datos
        for j in aux:
            auxb=j[:-1]
            column_name.remove(auxb)
            DatosDroped=DatosDroped.drop([auxb],axis=1)
        MatrizDatos=np.array(Datos[column_name])
        print(column_name)
        ArbolesAlgorithm()
        return redirect('/arboles_result.html/')
    else:
        print("PRUEBA DE ALCANCE")
        print(column_name)
        return render(request,'arboles_simulate_3.html',{
            'columnas':column_name,
        })
def ArbolesResults(request):
    global filename
    global Datos
    global estadoResultado
    global X_train_size
    global X_train
    global exactitud
    global Matriz_Clasificacion
    global column_name
    global ValoresPredic
    global varDep
    global typeTree
    global typeClass
    global nestimators
    global ClasificacionAD
    global ValoresMod1
    X_train_frame=pd.DataFrame(X_train)
    if request.method=='POST':
        AuxPred={}
        for x in column_name:
            AuxPred[x]=[float(request.POST[x])]
        Sujeto = pd.DataFrame(AuxPred)
        clasificado=ClasificacionAD.predict(Sujeto)
        estadoResultado="1"
        return render(request,'arboles_result.html',{
            'filename':filename,
            'data': Datos,
            'estado':estadoResultado,
            'Xsize':X_train_size,
            'exact':exactitud,
            'MatClas':Matriz_Clasificacion,
            'lista':column_name,
            'clase': varDep,
            'entrenamiento':X_train_frame,
            'clas':clasificado,
            'validation':ValoresMod1,
            'n':nestimators,
            'tip':typeClass,
            'tipo':typeTree,
        })
    else:
        estadoResultado="0"
        clasificado=0
        return render(request,'arboles_result.html',{
            'filename':filename,
            'data': Datos,
            'estado':estadoResultado,
            'Xsize':X_train_size,
            'exact':exactitud,
            'MatClas':Matriz_Clasificacion,
            'lista':column_name,
            'clase': varDep,
            'entrenamiento':X_train_frame,
            'clas':clasificado,
            'validation':ValoresMod1,
            'n':nestimators,
            'tip':typeClass,
            'tipo':typeTree,
        })

def ArbolesAlgorithm():
    global Datos
    global column_name
    global valoresClas
    global varDep
    global DatosDroped
    global X
    global Y
    global X_train
    global X_validation
    global Y_train
    global X_train_size
    global Y_train_size
    global Y_validation
    global size_train
    global exactitud
    global Matriz_Clasificacion
    global ValoresPredic
    global ClasificacionAD
    global maxDepth
    global minSampleLeaf
    global minSampleSplit
    global ValoresMod1
    global ImportanciaMod1
    global typeTree
    global typeClass
    global nestimators
    ValoresPredic=Datos[varDep].unique().tolist()
    #Datos=Datos.replace({ValoresPredic[0]: 0, ValoresPredic[1]: 1})
    X = np.array(DatosDroped[column_name])
    Y = np.array(Datos[[varDep]])
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size = float(size_train), 
                                                                random_state = 0,
                                                                shuffle = True)
    X_train_size=len(X_train)
    Y_train_size=len(Y_train)
    if typeTree=="tree":
        ClasificacionAD = DecisionTreeClassifier(max_depth=int(maxDepth), 
                                            min_samples_split=int(minSampleSplit), 
                                            min_samples_leaf=int(minSampleLeaf),
                                            random_state=1234)
    else:
        ClasificacionAD = RandomForestClassifier(n_estimators=int(nestimators),
                                            max_depth=int(maxDepth), 
                                            min_samples_split=int(minSampleSplit), 
                                            min_samples_leaf=int(minSampleLeaf),
                                            random_state=0)
    ClasificacionAD.fit(X_train, Y_train)
    Y_ClasificacionAD = ClasificacionAD.predict(X_validation)
    ValoresMod1 = pd.DataFrame(Y_validation, Y_ClasificacionAD)
    exactitud=accuracy_score(Y_validation, Y_ClasificacionAD)
    exactitud=exactitud*100
    exactitud=round(exactitud,3)
    ModeloClasificacion1 = ClasificacionAD.predict(X_validation)
    Matriz_Clasificacion = pd.crosstab(Y_validation.ravel(), 
                                   ModeloClasificacion1, 
                                   rownames=['Reales'], 
                                   colnames=['Clasificación']) 
    ImportanciaMod1 = pd.DataFrame({'Variable': list(Datos[column_name]),
                                'Importancia': ClasificacionAD.feature_importances_}).sort_values('Importancia', ascending=False)
    if typeTree=="tree":
        plt.figure(figsize=(16,16))  
        plot_tree(ClasificacionAD, feature_names = column_name)
        plt.savefig("./media/arbolGenerated.jpg", bbox_inches='tight')
    if typeClass=="sing":
        CurvaRoc= RocCurveDisplay.from_estimator(ClasificacionAD,
                                X_validation,
                                Y_validation,
                                name=varDep)
        plt.savefig("./media/ROC.jpg", bbox_inches='tight')

def imprimeMensaje():
    print("Prueba de alcance")