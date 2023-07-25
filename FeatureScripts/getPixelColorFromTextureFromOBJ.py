import PIL
from PIL import Image, ImageFile
from PIL import ImageFilter
#from PIL import GifImagePlugin
from psd_tools import PSDImage
import sys
import os
import re
import numpy as np
import copy
from scipy import spatial

sys.setrecursionlimit(10000)

#############################
# Input: texture folder, OBJ, PLY
# 1. Read the texture folder and get the pixels of images
#    Store in a dictionary {img_name: pixels}
# 2. Read the OBJ for face and vertices and image name for each face.
#    Store in a dictionary. {fid: {v: v1,v2,v3},{img: image_name}}
# 3. Read ply line - P
#    3.1 For each point P, get the face index F
#    3.2 Get the Vertices for the face F
#    3.3 Get the barycentric co-ordinate for the vertices and P - b_u, b_v, b_w
#    3.4 Get the P_u, P_v using b_u, b_v and b_w -> [(v1_u*b_u + v2_u*b_v + v3_u*b_w), ( v1_v*b_u + v2_v*b_v + v3_v*b_w)]
#    3.5 Read the pixel color using P_u and P_v

class Color:
    def __init__(self):
        self.ka = {"r":0.0, "g":0.0, "b":0.0, "a":0.0}
        self.kd = {"r": 0.0, "g":0.0, "b":0.0, "a":0.0}
        self.trans = {"r": 0.0, "g":0.0, "b":0.0, "a":0.0}
        self.transparency = 1.0
        self.tex = ""
        self.a = 1.0


class ColorFromTexture:
    def __init__(self, filename, OBJFolder, TextureFolder, PLYFolder, MTLFolder, faceindexFolder):
        self.vertices = []
        self.textures = []
        self.bary_coord = []
        self.face = {}
        self.faceid= 0
        self.imgToPixel = {}
        self.imgFiles = {}
        self.materialToColor = {}
        self.P_color = []
        self.P_nearest = {}
        self.P = []


        self.OBJFolder = OBJFolder
        self.TexFolder = TextureFolder
        self.PLYFolder = PLYFolder
        self.MTLFolder = MTLFolder
        self.faceindexFolder = faceindexFolder
        self.fname = filename

        self.avgDistance = 0.0

        self.initializeMeta()

    def initializeMeta(self):
        self.readPixelsFromImage()
        self.readMTLInfo()
        self.getFaceInfo()
        self.collectMetadataOfPoints()

    def run(self):
        self.findNearestNeighbourPoint(numofneigh=2)
        self.getColorForPoints()

    def readPixelsFromImage(self):
        imagepath = os.path.join(self.TexFolder, self.fname)

        if os.path.exists(imagepath):
            files = os.listdir(imagepath)

            for imgfile in files:
                imgfilepath = os.path.join(imagepath, imgfile)
                self.imgFiles[imgfile[0:-4]] = imgfilepath
                self.getAllMipMaps(imgfilepath, imgfile[0:-4])

    def readMTLInfo(self):
        MTLfile = os.path.join(self.MTLFolder, self.fname+".mtl")

        if os.path.exists(MTLfile):
            fread = open(MTLfile, 'r')
            c = Color()
            matname = ""
            for line in fread:
                line = line.strip()

                if len(line):
                    if line.startswith("newmtl "):
                        if len(matname):
                            self.materialToColor[matname] = copy.deepcopy(c)
                            c = Color()
                        matname = line.split()[1]
                    if line.startswith("Kd "):
                        data = line.split()
                        c.kd["r"] = float(data[1])
                        c.kd["g"] = float(data[2])
                        c.kd["b"] = float(data[3])
                        c.kd["a"] = float(data[4])
                    if line.startswith("Ka "):
                        data = line.split()
                        c.ka["r"] = float(data[1])
                        c.ka["g"] = float(data[2])
                        c.ka["b"] = float(data[3])
                        c.ka["a"] = float(data[4])
#                    if line.startswith("Ks "):
#                        data = line.split()
#                        ks.r = (float)data[1]
#                        ks.g = (float)data[2]
#                        ks.b = (float)data[3]
#                    if line.startswith("Ke "):
#                        data = line.split()
#                        ke.r = (float)data[1]
#                        ke.g = (float)data[2]
#                        ke.b = (float)data[3]
#                    if line.startswith("Ns "):
#                        ns = (float)(line.split()[1])
                    if line.startswith("d "):
                        c.a = float(line.split()[1])
                    if line.startswith("trans "):
                        data = line.split()
                        c.trans["r"] = float(data[1])
                        c.trans["g"] = float(data[2])
                        c.trans["b"] = float(data[3])
                        c.trans["a"] = float(data[4])
#                    if line.startswith("Ni "):
#                        ni = (float)(line.split()[1])
                    if line.startswith("map_Kd "):
                        line = line.split()[1]
                        if os.path.exists(os.path.join(self.TexFolder, line)):
                            line = line.split("/")[1]
                            c.tex = line[0:-4]
            if len(matname):
                self.materialToColor[matname] = copy.deepcopy(c)

    def getFaceInfo(self):
        fread = open(os.path.join(self.OBJFolder, self.fname+".obj"))
        material = ""
        faceid = 0
        for line in fread:
            line = line.strip()
            if line.startswith("v "):
                data = line.split()
                self.vertices.append([float(data[1].strip()), float(data[2].strip()), float(data[3].strip())])
            elif line.startswith("usemtl"):
                material = line.split(" ")[1]
            elif line.startswith("vt "):
                data = line.split()
                self.textures.append([float(data[1]), float(data[2])])
            elif line.startswith("f "):
                if faceid == 2:
                    print(line)
                    print(material)
                vertex_info = line.split()
                if len(vertex_info) < 3:
                    print("issue with triangle")
                    exit

                v1 = vertex_info[1].split('/')
                v2 = vertex_info[2].split('/')
                v3 = vertex_info[3].split('/')

                vertex = [int(v1[0])-1, int(v2[0])-1, int(v3[0])-1]
                texture = []
                if len(v1[1]):
                    texture = [int(v1[1])-1, int(v2[1])-1, int(v3[1])-1]
                self.face[faceid] = {"v":vertex, "tex":texture, "mat":material}
                faceid += 1

    def areaOfTriangle(self, v1, v2, v3):
        a =  (v2 - v1)
        b =  (v3 - v1)

        c = np.cross(a,b)
        area = float(np.linalg.norm(c))/float(2.0)
        return area

    def collectMetadataOfPoints(self):
        fread = open(os.path.join(self.PLYFolder, self.fname+".ply"),"r")
        faceindexfile = open(os.path.join(self.faceindexFolder, self.fname+".txt"),"r")

        faceindex = []
        for index in faceindexfile:
            index = index.strip()
            faceindex.append(int(index))
        faceindexfile.close()

        count = 0
        index = 0
        for line in fread:
            #print(line)
            if count < 12:
                count += 1
                continue
            line = line.strip()
            data = line.split()
            #face_id = data[len(data)-1]
            face_id = faceindex[index]
            print("index = {}".format(index))
            index += 1
            P = np.array([float(data[0]), float(data[1]), float(data[2])])

            vertex = self.face[int(face_id)]["v"]
            vertex = [np.array(self.vertices[vertex[0]]), np.array(self.vertices[vertex[1]]), np.array(self.vertices[vertex[2]])]
            texture = self.face[int(face_id)]["tex"]
            material = self.face[int(face_id)]["mat"]
            print("material")
            print(material)
            if len(material) <= 0:
                print(face_id)
                print(texture)
            #print("material = {}".format(material))
            area_ABC = self.areaOfTriangle(vertex[0], vertex[1], vertex[2])
            if(area_ABC <= 0):
                #print(area_ABC)
                #print(P)
                area_ABC = 1e-10
                #print(face_id)
                #print(self.face[int(face_id)])
                #return

            
            #area_PBC
            x = (self.areaOfTriangle(P, vertex[1], vertex[2]))/area_ABC
            y = (self.areaOfTriangle(P, vertex[2], vertex[0]))/area_ABC
            #z = (self.areaOfTriangle(P, vertex[0], vertex[1]))/area_ABC
            z = 1 - x - y
            if x >= 1 or y >=1 or z >= 1 or x <0 or y < 0 or z < 0:
                x = 0.3
                y = 0.3
                z = 0.4
            #else:
#            if y >= 1:
#                print(area_ABC)
#                print(x)
#                print(y)
#                print(z)
#                print(P)
#                print(vertex)
#            z = (1-x-y)#(self.areaOfTriangle(P, vertex[0], vertex[1]))/area_ABC

            self.P.append({"p":P, "v":vertex,"tex":texture,"mat":material, "bary":[x,y,z]})

    def findNearestNeighbourPoint(self, numofneigh=2):
        points = []
        for v in self.P:
            points.append(v["p"])

        tree = spatial.KDTree(points)
        #print(tree.data)
        self.avgDistance = 0.0
        count = 0
        for p in self.P:
            
            distances, index = tree.query(p["p"], k=numofneigh)
            i = 1
            if distances[1] == 0:
                while distances[i] == 0 or i < 15:
                    distances, index = tree.query(p["p"], k=(numofneigh+i))
                    i += 1
                    if distances[i] > 0:
                        break
                #print(distances)
                #print(index)

            self.avgDistance += distances[i]
            #if not index[1]:
            #    self.P_nearest[index[0]] = index[0]
            #else:
            self.P_nearest[count] = index[i]
            count += 1
        #print(self.P_nearest)

        self.avgDistance = 0.5*self.avgDistance / len(self.P)
        #print("********** Avg distance *************")
        #print(self.avgDistance)

    def getTexCoordForP(self, bary_coord, texture_info):
        #print(texture_info)
        v1_tex = np.array(self.textures[texture_info[0]])
        v1_tex[v1_tex < 0] = 1 - v1_tex[v1_tex < 0]
        v1_tex = v1_tex % 1

        v2_tex = np.array(self.textures[texture_info[1]])
        v2_tex[v2_tex < 0] = 1 - v2_tex[v2_tex < 0]
        v2_tex = v2_tex % 1
        
        v3_tex = np.array(self.textures[texture_info[2]])
        v3_tex[v3_tex < 0] = 1 - v3_tex[v3_tex < 0]
        v3_tex = v3_tex % 1

#        print(v1_tex)
#        print(v2_tex)
#        print(v3_tex)
#        print(bary_coord)


        P_U = bary_coord[0]*(v1_tex[0]) + bary_coord[1]*(v2_tex[0]) + bary_coord[2]*(v3_tex[0])
        P_V = bary_coord[0]*(v1_tex[1]) + bary_coord[1]*(v2_tex[1]) + bary_coord[2]*(v3_tex[1])

#        print(P_U)
#        print(P_V)

        return [P_U, P_V]

    def createMipMap(self, img, imgfilename, width, height):
        if width <= 1 or height <= 1:
            return self.imgToPixel[imgfilename]

        img_downsize = img.resize((width,height), resample=PIL.Image.LANCZOS)
        img_downsize = img_downsize.filter(ImageFilter.BLUR)
        pix = img_downsize.load()
        self.imgToPixel[imgfilename].append({"w":width, "h":height, "pix":pix})
        width = width // 2
        height = height // 2
        self.imgToPixel[imgfilename] = self.createMipMap(img_downsize, imgfilename, width, height)
        return self.imgToPixel[imgfilename]

    def resizeImage(self, imgname, width, height):
        img = Image.open(self.imgFiles[imgname])
        pix = img.load()
        img_resize = img.resize((width,height), resample=PIL.Image.LANCZOS)
        return img_resize
        #self.imgToPixel[imgfilename].append({"w":width, "h":height, "pix":pix})
   
    def getAllMipMaps(self, imgfilepath, imgfilename):
        #print(imgfilepath)
        fname, ext = os.path.splitext(imgfilepath)
        if ext == '.psd':
            img = PSDImage.load(imgfilepath).as_PIL()
            img = img.convert('RGB')
        else:
            try:
                img = Image.open(imgfilepath)
            except:
                img = Image.new('RGB', (256,256), (200,200,200)) 


        ImageFile.LOAD_TRUNCATED_IMAGES = True
        #else:
        if ext == '.gif' or ext == '.GIF':
            img = img.convert('RGB')
            
        img = img.filter(ImageFilter.BLUR)
        pix = img.load()
        self.imgToPixel[imgfilename] = []
        self.imgToPixel[imgfilename].append({"w":img.width, "h":img.height, "pix":pix})
        width = img.width //2
        height = img.height // 2
        self.imgToPixel[imgfilename] = self.createMipMap(img, imgfilename,width, height)

    def getMipMapIndex(self, img_width, img_height):
        diff_in_texel = [img_width*self.avgDistance, img_height*self.avgDistance]

        mindiff = min(diff_in_texel)
        minside = min(img_width, img_height)
        #print(diff_in_texel)
        
        #print(mindiff)
        #print(minside)
        #print(img_width)
        #print(img_height)

        k = 0
        if minside > 1:
            while True:
                minside = minside //2
                if minside <= 1 or minside < mindiff:
                    break
                k += 1
                if k > 15:
                    break
        return k


    def getMipMapForP(self,Pindex):
        nearestP = self.P_nearest[Pindex]
        P_material = self.P[Pindex]["mat"]
        Pneigh_material = self.P[nearestP]["mat"]
        print("p material")
        print(P_material)
        P_info = self.materialToColor[P_material]
        P_tex = P_info.tex
        Pneigh_info = self.materialToColor[Pneigh_material]
        Pneigh_tex = Pneigh_info.tex
        #print(self.P[nearestP])

        if len(P_tex):# self.P[Pindex]['tex']:
            P_U, P_V = self.getTexCoordForP(self.P[Pindex]["bary"], self.P[Pindex]["tex"])
            if len(Pneigh_tex): #self.P[nearestP]['tex']:
                Pneigh_U, Pneigh_V = self.getTexCoordForP(self.P[nearestP]["bary"], self.P[nearestP]["tex"])
                if P_material == Pneigh_material:
                    #print(P_material)
                    #print(self.imgToPixel[P_material])
                    #info = self.materialToColor[P_material]
                    #P_tex = info.tex
                    img_width = self.imgToPixel[P_tex][0]["w"]
                    img_height = self.imgToPixel[P_tex][0]["h"]

                    mipmapindex = self.getMipMapIndex(img_width, img_height)
                    #mipmapindex = 0
                    pix = self.imgToPixel[P_tex][mipmapindex]["pix"]
                    img_resize_width = self.imgToPixel[P_tex][mipmapindex]["w"]
                    img_resize_height = self.imgToPixel[P_tex][mipmapindex]["h"]
                    
                    self.getColorOfPointByInterpolation(P_material, Pneigh_material, True, img_resize_width, img_resize_height, img_resize_width, img_resize_height, pix, pix, P_U, P_V, Pneigh_U, Pneigh_V)
                    #self.getColorOfPointByInterpolation(P_material,Pneigh_material, True, img_width, img_height, img_width, img_height, pix, pix, P_U, P_V, Pneigh_U, Pneigh_V)
                else:
                    #P_info = self.materialToColor[P_material]
                    #P_tex = P_info.tex
                    P_img_width = self.imgToPixel[P_tex][0]["w"]
                    P_img_height = self.imgToPixel[P_tex][0]["h"]
                    P_mipmapindex = self.getMipMapIndex(P_img_width, P_img_height)

                    #Pneigh_info = self.materialToColor[Pneigh_material]
                    #Pneigh_tex = Pneigh_info.tex
                    Pneigh_img_width = self.imgToPixel[Pneigh_tex][0]["w"]
                    Pneigh_img_height = self.imgToPixel[Pneigh_tex][0]["h"]
                    Pneigh_mipmapindex = self.getMipMapIndex(Pneigh_img_width, Pneigh_img_height)

                    #P_mipmapindex = 0
                    #Pneigh_mipmapindex = 0

#                    print(P_mipmapindex)
#                    print(P_tex)
#                    print(self.imgToPixel[P_tex])
#                    print(P_img_width)
#                    print(P_img_height)

                    #print(P_tex)
                    #print(Pneigh_tex)
                   
                    P_pix = self.imgToPixel[P_tex][P_mipmapindex]["pix"]
                    Pneigh_pix = self.imgToPixel[Pneigh_tex][Pneigh_mipmapindex]["pix"]
                    Pimg_resize_width = self.imgToPixel[P_tex][P_mipmapindex]["w"]
                    Pimg_resize_height = self.imgToPixel[P_tex][P_mipmapindex]["h"]
                    Pneighimg_resize_width = self.imgToPixel[Pneigh_tex][Pneigh_mipmapindex]["w"]
                    Pneighimg_resize_height = self.imgToPixel[Pneigh_tex][Pneigh_mipmapindex]["h"]

                    #self.getColorOfPointByInterpolation(P_material,Pneigh_material, True, P_img_width, P_img_height, Pneigh_img_width, Pneigh_img_height, P_pix, Pneigh_pix, P_U, P_V, Pneigh_U, Pneigh_V)
                    self.getColorOfPointByInterpolation(P_material,Pneigh_material, True, Pimg_resize_width, Pimg_resize_height, Pneighimg_resize_width, Pneighimg_resize_height, P_pix, Pneigh_pix, P_U, P_V, Pneigh_U, Pneigh_V)
            else:
                #P_info = self.materialToColor[P_material]
                #P_tex = P_info.tex
                img_width = self.imgToPixel[P_tex][0]["w"]
                img_height = self.imgToPixel[P_tex][0]["h"]

                mipmapindex = self.getMipMapIndex(img_width, img_height)
                #print(mipmapindex)
                #mipmapindex = 0
                pix = self.imgToPixel[P_tex][mipmapindex]["pix"]
                img_resize_width = self.imgToPixel[P_tex][mipmapindex]["w"]
                img_resize_height = self.imgToPixel[P_tex][mipmapindex]["h"]
                #print(img_resize_width)
                #print(img_resize_height)
                self.getColorOfPointByInterpolation(P_material,Pneigh_material, True, img_resize_width, img_resize_height, 0,0, pix, [], P_U, P_V)
        else:
            if len(Pneigh_tex): #self.P[nearestP]['tex']:
                #print(Pneigh_tex)
                Pneigh_U, Pneigh_V = self.getTexCoordForP(self.P[nearestP]["bary"], self.P[nearestP]["tex"])
                #Pneigh_info = self.materialToColor[Pneigh_material]
                #Pneigh_tex = Pneigh_info.tex
                Pneigh_img_width = self.imgToPixel[Pneigh_tex][0]["w"]
                Pneigh_img_height = self.imgToPixel[Pneigh_tex][0]["h"]
                Pneigh_mipmapindex = self.getMipMapIndex(Pneigh_img_width, Pneigh_img_height)

                #Pneigh_mipmapindex = 0
                Pneigh_pix = self.imgToPixel[Pneigh_tex][Pneigh_mipmapindex]["pix"]
                Pneighimg_resize_width = self.imgToPixel[Pneigh_tex][Pneigh_mipmapindex]["w"]
                Pneighimg_resize_height = self.imgToPixel[Pneigh_tex][Pneigh_mipmapindex]["h"]

                #self.getColorOfPointByInterpolation(P_material,Pneigh_material, False, 0,0, Pneigh_img_width, Pneigh_img_height, [], Pneigh_pix, None,None, Pneigh_U, Pneigh_V)
                self.getColorOfPointByInterpolation(P_material,Pneigh_material, False, 0,0, Pneighimg_resize_width, Pneighimg_resize_height, [], Pneigh_pix, None,None, Pneigh_U, Pneigh_V)
            else:
                self.getColorOfPointByInterpolation(P_material,Pneigh_material)

    def getAvgColorForP(self,Pindex):
        nearestP = self.P_nearest[Pindex][0]
        #print(str(Pindex)+"::"+str(nearestP))
        #print(self.P[Pindex])
        #print(self.P[nearestP])
        P_material = self.P[Pindex]["mat"]
        P_c_info = self.materialToColor[P_material]
        Pneigh_material = self.P[nearestP]["mat"]
        Pneigh_c_info = self.materialToColor[Pneigh_material]
        
        if self.P[Pindex]['tex'] and self.P[nearestP]['tex']:
            P_U, P_V = self.getTexCoordForP(self.P[Pindex]["bary"], self.P[Pindex]["tex"])
            Pneigh_U, Pneigh_V = self.getTexCoordForP(self.P[nearestP]["bary"], self.P[nearestP]["tex"])
            img_width = self.imgToPixel[material][0]["w"]
            img_height = self.imgToPixel[material][0]["h"]
            pix = self.imgToPixel[material][0]["pix"]

            P_row = img_width*P_U
            P_col = img_height*P_V
            Pneigh_row = img_width*Pneigh_U
            Pneigh_col = img_height*Pneigh_V

            distance = np.sqrt(np.square(P_row - Pneigh_row) + np.square(P_col - Pneigh_col))
            radius = int(np.ceil(distance / 2))
            radius = radius if radius > 0 else 1


            P_color = self.getColorOfPointByAverage(radius, pix, int(np.floor(P_row)), int(np.floor(P_col)), img_width, img_height)
            P_color[0] = 1.0 * (1.0 - c_info.trans["r"] * c_info.transparency ) + P_color[0] * (c_info.trans["r"])* c_info.transparency          
            P_color[1] = 1.0 * (1.0 - c_info.trans["g"] * c_info.transparency ) + P_color[1] * (c_info.trans["g"])* c_info.transparency          
            P_color[2] = 1.0 * (1.0 - c_info.trans["b"] * c_info.transparency ) + P_color[2] * (c_info.trans["b"])* c_info.transparency          
            P_color = [P_color[0]*c_info.kd["r"] , P_color[1]*c_info.kd["g"] , P_color[2]*c_info.kd["b"], int(c_info.a * 255)]
        else:
            P_color = [int((c_info.kd["r"])*255) ,int((c_info.kd["g"]) *255),int((c_info.kd["b"])*255), int(c_info.a*255)]

        self.P_color.append([int(P_color[0]), int(P_color[1]), int(P_color[2]), int(P_color[3])])#int( c_info.a*255)])

    def pixelColor(self, material, img_width, img_height, P_U, P_V, pix):
        c_info = self.materialToColor[material]
        
        pixel_row = img_width*P_U
        pixel_col = img_height*P_V
        
        pixel_row_1 = int(np.floor(pixel_row))
        pixel_row_2 = 0 if np.ceil(pixel_row) >= img_width else int(np.ceil(pixel_row))

        pixel_col_1 = int(np.floor(pixel_col))
        pixel_col_2 = 0 if np.ceil(pixel_col) >= img_height else int(np.ceil(pixel_col))

        # Interpolate the color of the texture
        row_interpolate = pixel_row % 1
        col_interpolate = pixel_col % 1

#        print("***********")
#        print(P_U)
#        print(P_V)
#        print(pixel_row_1)
#        print(pixel_row_2)
#        print(pixel_col_1)
#        print(pixel_col_2)
#        print(pix[pixel_row_1, pixel_col_1])
#        print(pix[pixel_row_2, pixel_col_1])
#        print(material)


        color_row_1 = row_interpolate*np.array(pix[pixel_row_1,pixel_col_1])+ (1.0-row_interpolate)*np.array(pix[pixel_row_2,pixel_col_1])
        color_row_2 = row_interpolate*np.array(pix[pixel_row_1,pixel_col_2])+ (1.0-row_interpolate)*np.array(pix[pixel_row_2,pixel_col_2])

        P_color = (col_interpolate*color_row_1 + (1.0-col_interpolate)*color_row_2)

        P_color[0] = 1.0 * (1.0 - c_info.trans["r"] * c_info.transparency ) + P_color[0] * (c_info.trans["r"])* c_info.transparency          
        P_color[1] = 1.0 * (1.0 - c_info.trans["g"] * c_info.transparency ) + P_color[1] * (c_info.trans["g"])* c_info.transparency          
        P_color[2] = 1.0 * (1.0 - c_info.trans["b"] * c_info.transparency ) + P_color[2] * (c_info.trans["b"])* c_info.transparency          
        #P_color[3] = c_info.a #1.0 * (1.0 -c_info.a) + ( c_info.a)* P_color[3]     
        #P_color = [c_info.ka["r"] + P_color[0]*c_info.kd["r"] , c_info.ka["g"] + P_color[1]*c_info.kd["g"] , c_info.ka["b"] + P_color[2]*c_info.kd["b"], int(c_info.a * 255)]
        P_color = [P_color[0]*c_info.kd["r"] , P_color[1]*c_info.kd["g"] , P_color[2]*c_info.kd["b"], int(c_info.a * 255)]
        return P_color

    def getColorOfPointByInterpolation(self,Pmaterial,Pneigh_material, texture=False, img_width=0, img_height=0, Pneigh_img_width=0,Pneigh_img_height=0, P_pix=None,Pneigh_pix=None, P_U=None, P_V=None, Pneigh_U=None, Pneigh_V=None):
        if texture:
            if P_pix and Pneigh_pix:
                P_color = self.pixelColor(Pmaterial, img_width, img_height, P_U, P_V, P_pix)
                Pneigh_color = self.pixelColor(Pneigh_material, Pneigh_img_width, Pneigh_img_height, Pneigh_U, Pneigh_V, Pneigh_pix)
            elif P_pix:
                P_color = self.pixelColor(Pmaterial, img_width, img_height, P_U, P_V, P_pix)
                c_info = self.materialToColor[Pneigh_material]
                Pneigh_color = [int((c_info.kd["r"])*255) ,int((c_info.kd["g"]) *255),int((c_info.kd["b"])*255), int(c_info.a*255)]

        else:
            if Pneigh_pix:
                c_info = self.materialToColor[Pmaterial]
                P_color = [int((c_info.kd["r"])*255) ,int((c_info.kd["g"]) *255),int((c_info.kd["b"])*255), int(c_info.a*255)]
                Pneigh_color = self.pixelColor(Pneigh_material, img_width, img_height, Pneigh_U, Pneigh_V, Pneigh_pix)
            else:
                c_info = self.materialToColor[Pmaterial]
                P_color = [int((c_info.kd["r"])*255) ,int((c_info.kd["g"]) *255),int((c_info.kd["b"])*255), int(c_info.a*255)]
                c_info = self.materialToColor[Pneigh_material]
                Pneigh_color = [int((c_info.kd["r"])*255) ,int((c_info.kd["g"]) *255),int((c_info.kd["b"])*255), int(c_info.a*255)]

        color = (np.array(P_color)+ np.array(Pneigh_color))/2
        #print(color)
        self.P_color.append([int(color[0]), int(color[1]), int(color[2]), int(color[3])])#int( c_info.a*255)])
        #self.P_color.append([int(P_color[0]), int(P_color[1]), int(P_color[2]), int(P_color[3])])#int( c_info.a*255)])


    def getColorOfPointByAverage(self, radius, pix, pixel_row, pixel_col, img_width, img_height):
        #print(radius)
        final_pixel_color = np.array([0,0,0])
        for i in range(-radius, radius):
            for j in range(-radius, radius):
                r = pixel_row + i
                if r < 0:
                    r = img_width-1 + r
                if r >= img_width:
                    r = r - img_width
                c = pixel_col + j
                if c < 0 :
                    c = img_height-1 +c
                if c >= img_height:
                    c = c - img_height
                #if r < 0 or c < 0 or r >= img_width or c >= img_height:
                #    continue
                final_pixel_color = final_pixel_color + np.array(pix[r,c])

        final_pixel_color = final_pixel_color / ((2*radius+1) * (2*radius+1))
        #print(final_pixel_color)
        return final_pixel_color

    def getColorForPoints(self):
        n = len(self.P)
        sumv = 0
        for i in range(n):
            self.getMipMapForP(i)

    def writePLY(self, destFolder):
        fread = open(os.path.join(self.PLYFolder, self.fname+".ply"),'r')
        f = open(os.path.join(destFolder, self.fname+'.ply'),'w')
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("element vertex 100000\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property float nx\n")
        f.write("property float ny\n")
        f.write("property float nz\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("property uchar alpha\n")
        f.write("element face 0\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        index= 0
        count = 0
        #print(len(self.P_color))
        for line in fread:
            if count < 12:
                count += 1
                continue
            line = line.strip()
            #line = line.rsplit(' ',1)[0]
            #line = line.strip()
            #print(self.P_color)
            #f.write(line +" "+str(self.P_color[index][0])+" "+str(self.P_color[index][1])+" "+str(self.P_color[index][2])+" "+str(self.P_color[index][3])+"\n")
            f.write(line +" "+str(self.P_color[index][0]/255.0)+" "+str(self.P_color[index][1]/255.0)+" "+str(self.P_color[index][2]/255.0)+" "+str(self.P_color[index][3]/255.0)+"\n")
            index += 1
            #print(index)
        f.close()




def getColorFromTextureOfAFile(filename, objFolder, textureFolder, PLYFolder, MTLFolder, faceindexFolder, destinationFolder):
    print(filename)
    ctex = ColorFromTexture(filename, objFolder, textureFolder, PLYFolder, MTLFolder, faceindexFolder)
    #print(ctex)
    ctex.run()
    ctex.writePLY(destinationFolder)

def getColorFromTextureFromAList(inputfile, objFolder, textureFolder, PLYFolder, MTLFolder, faceindexFolder, destinationFolder):
    fread = open(inputfile, 'r')
    for line in fread:
        line = line.strip()
        getColorFromTextureOfAFile(line, objFolder, textureFolder, PLYFolder, MTLFolder, faceindexFolder, destinationFolder)
    fread.close()

if __name__ == '__main__':
    param = sys.argv[1]
    if param == 'file':
        getColorFromTextureOfAFile(*sys.argv[2:])
    elif param == 'list':
        getColorFromTextureFromAList(*sys.argv[2:])

