### Compute the IOU of two rotated 2D rectangle

import math
import numpy as np
import sys
import random
import torch

from torch.autograd import Function
import torch.nn as nn

#from compute_ious import compute_ious_whih_shapely
from scipy.spatial import ConvexHull


## This function is used to determine whether a point is inside a rectangle or not
class compute_vertex(Function):
    '''
    Compute the corners which are inside the rectangles
    '''

    @staticmethod
    def forward(ctx, corners_gboxes, corners_qboxes):

        np_corners_gboxes = corners_gboxes.cpu().numpy()
        np_corners_qboxes = corners_qboxes.cpu().detach().numpy()
        N = corners_gboxes.shape[0]
        num_of_intersections = np.zeros((N,), dtype=np.int32)
        intersections = np.zeros((N, 16), dtype=np.float32)
        flags_qboxes  = np.zeros((N, 4), dtype=np.float32)
        flags_gboxes  = np.zeros((N, 4), dtype=np.float32)
        flags_inters  = np.zeros((N, 4, 4), dtype=np.float32)

        for iter in range(N):
            # step 1: determine how many corners from corners_gboxes inside the np_qboxes
            ab0 = np_corners_qboxes[iter, 2] - np_corners_qboxes[iter, 0]
            ab1 = np_corners_qboxes[iter, 3] - np_corners_qboxes[iter, 1]
            ad0 = np_corners_qboxes[iter, 6] - np_corners_qboxes[iter, 0]
            ad1 = np_corners_qboxes[iter, 7] - np_corners_qboxes[iter, 1]
            for i in range(4):
                ap0 = np_corners_gboxes[iter, i * 2] - np_corners_qboxes[iter, 0]
                ap1 = np_corners_gboxes[iter, i * 2 + 1] - np_corners_qboxes[iter, 1]
                abab = ab0 * ab0 + ab1 * ab1
                abap = ab0 * ap0 + ab1 * ap1
                adad = ad0 * ad0 + ad1 * ad1
                adap = ad0 * ap0 + ad1 * ap1
                if (abab >= abap and abap >= 0 and adad >= adap and adap >= 0):
                    intersections[iter, num_of_intersections[iter] * 2] = np_corners_gboxes[iter, i * 2]
                    intersections[iter, num_of_intersections[iter] * 2 + 1] = np_corners_gboxes[iter, i * 2 + 1]
                    num_of_intersections[iter] += 1
                    flags_gboxes[iter, i] = 1.0

            # step 2: determine how many corners from np_qboxes inside corners_gboxes
            ab0 = np_corners_gboxes[iter, 2] - np_corners_gboxes[iter, 0]
            ab1 = np_corners_gboxes[iter, 3] - np_corners_gboxes[iter, 1]
            ad0 = np_corners_gboxes[iter, 6] - np_corners_gboxes[iter, 0]
            ad1 = np_corners_gboxes[iter, 7] - np_corners_gboxes[iter, 1]
            for i in range(4):
                ap0 = np_corners_qboxes[iter, i * 2] - np_corners_gboxes[iter, 0]
                ap1 = np_corners_qboxes[iter, i * 2 + 1] - np_corners_gboxes[iter, 1]
                abab = ab0 * ab0 + ab1 * ab1
                abap = ab0 * ap0 + ab1 * ap1
                adad = ad0 * ad0 + ad1 * ad1
                adap = ad0 * ap0 + ad1 * ap1
                if (abab >= abap and abap >= 0 and adad >= adap and adap >= 0):
                    intersections[iter, num_of_intersections[iter] * 2] = np_corners_qboxes[iter, i * 2]
                    intersections[iter, num_of_intersections[iter] * 2 + 1] = np_corners_qboxes[iter, i * 2 + 1]
                    num_of_intersections[iter] += 1
                    flags_qboxes[iter, i] = 1.0

            # step 3: find the intersection of all the edges
            for i in range(4):
                for j in range(4):
                    A = np.zeros((2,), dtype=np.float32)
                    B = np.zeros((2,), dtype=np.float32)
                    C = np.zeros((2,), dtype=np.float32)
                    D = np.zeros((2,), dtype=np.float32)

                    A[0] = np_corners_gboxes[iter, 2 * i]
                    A[1] = np_corners_gboxes[iter, 2 * i + 1]
                    B[0] = np_corners_gboxes[iter, 2 * ((i + 1) % 4)]
                    B[1] = np_corners_gboxes[iter, 2 * ((i + 1) % 4) + 1]

                    C[0] = np_corners_qboxes[iter, 2 * j]
                    C[1] = np_corners_qboxes[iter, 2 * j + 1]
                    D[0] = np_corners_qboxes[iter, 2 * ((j + 1) % 4)]
                    D[1] = np_corners_qboxes[iter, 2 * ((j + 1) % 4) + 1]

                    BA0 = B[0] - A[0]
                    BA1 = B[1] - A[1]
                    CA0 = C[0] - A[0]
                    CA1 = C[1] - A[1]
                    DA0 = D[0] - A[0]
                    DA1 = D[1] - A[1]

                    acd = DA1 * CA0 > CA1 * DA0
                    bcd = (D[1] - B[1]) * (C[0] - B[0]) > (C[1] - B[1]) * (D[0] - B[0])
                    if acd != bcd:
                        abc = CA1 * BA0 > BA1 * CA0
                        abd = DA1 * BA0 > BA1 * DA0
                        if abc != abd:
                            DC0 = D[0] - C[0]
                            DC1 = D[1] - C[1]
                            ABBA = A[0] * B[1] - B[0] * A[1]
                            CDDC = C[0] * D[1] - D[0] * C[1]
                            DH = BA1 * DC0 - BA0 * DC1
                            Dx = ABBA * DC0 - BA0 * CDDC
                            Dy = ABBA * DC1 - BA1 * CDDC
                            # DH = (B[1] - A[1]) * (D[0] - C[0]) - (B[0] - A[0]) * (D[1] - C[1])
                            # Dx = (A[0] * B[1] - B[0] * A[1]) * (D[0] - C[0]) - (B[0] - A[0]) * (C[0] * D[1] - D[0] * C[1])
                            # Dy = (A[0] * B[1] - B[0] * A[1]) * (D[1] - C[1]) - (B[1] - A[1]) * (C[0] * D[1] - D[0] * C[1])
                            if (num_of_intersections[iter] > 7):
                                print("iter = ", iter)
                                print("(%.4f %.4f) (%.4f %.4f) (%.4f %.4f) (%.4f %.4f)" % (
                                np_corners_gboxes[iter, 0], np_corners_gboxes[iter, 1],
                                np_corners_gboxes[iter, 2], np_corners_gboxes[iter, 3],
                                np_corners_gboxes[iter, 4], np_corners_gboxes[iter, 5],
                                np_corners_gboxes[iter, 6], np_corners_gboxes[iter, 7]))
                                print("(%.4f %.4f) (%.4f %.4f) (%.4f %.4f) (%.4f %.4f)" % (
                                np_corners_qboxes[iter, 0], np_corners_qboxes[iter, 1],
                                np_corners_qboxes[iter, 2], np_corners_qboxes[iter, 3],
                                np_corners_qboxes[iter, 4], np_corners_qboxes[iter, 5],
                                np_corners_qboxes[iter, 6], np_corners_qboxes[iter, 7]))
                                continue

                            intersections[iter, num_of_intersections[iter] * 2] = Dx / DH
                            intersections[iter, num_of_intersections[iter] * 2 + 1] = Dy / DH
                            num_of_intersections[iter] += 1
                            flags_inters[iter, i, j] = 1.0

        ctx.save_for_backward(corners_qboxes)
        ctx.corners_gboxes = corners_gboxes
        ctx.flags_qboxes = flags_qboxes
        ctx.flags_gboxes = flags_gboxes
        ctx.flags_inters = flags_inters
        # conver numpy to tensor
        tensor_intersections = torch.from_numpy(intersections)
        tensor_num_of_intersections = torch.from_numpy(num_of_intersections)

        return tensor_intersections, tensor_num_of_intersections

    @staticmethod
    def backward(ctx, *grad_outputs):

        _variables = ctx.saved_tensors
        corners_qboxes = _variables[0]
        corners_gboxes = ctx.corners_gboxes
        flags_qboxes = ctx.flags_qboxes
        flags_gboxes = ctx.flags_gboxes
        flags_inters = ctx.flags_inters
        grad_output = grad_outputs[0]

        np_corners_gboxes = corners_gboxes.cpu().numpy()
        np_corners_qboxes = corners_qboxes.cpu().detach().numpy()

        N = flags_qboxes.shape[0]
        n_of_inter = np.zeros((N,), dtype=np.int32)

        ### Check whether here is correct or not
        Jacbian_qboxes = np.zeros((N, 8, 16), dtype=np.float32)
        Jacbian_gboxes = np.zeros((N, 8, 16), dtype=np.float32)

        for iter in range(N):

            for i in range(4):
                if (flags_gboxes[iter, i] > 0):
                    Jacbian_gboxes[iter, i * 2, n_of_inter[iter] * 2] += 1.0
                    Jacbian_gboxes[iter, i * 2 + 1, n_of_inter[iter] * 2 + 1] += 1.0
                    n_of_inter[iter] += 1

            for i in range(4):
                if (flags_qboxes[iter, i] > 0):
                    Jacbian_qboxes[iter, i * 2, n_of_inter[iter] * 2] += 1.0
                    Jacbian_qboxes[iter, i * 2 + 1, n_of_inter[iter] * 2 + 1] += 1.0
                    n_of_inter[iter] += 1

            for i in range(4):
                for j in range(4):
                    if (flags_inters[iter, i, j] > 0):
                        ###
                        A = np.zeros((2,), dtype=np.float32)
                        B = np.zeros((2,), dtype=np.float32)
                        C = np.zeros((2,), dtype=np.float32)
                        D = np.zeros((2,), dtype=np.float32)
                        A[0] = np_corners_gboxes[iter, 2 * i]
                        A[1] = np_corners_gboxes[iter, 2 * i + 1]

                        B[0] = np_corners_gboxes[iter, 2 * ((i + 1) % 4)]
                        B[1] = np_corners_gboxes[iter, 2 * ((i + 1) % 4) + 1]

                        C[0] = np_corners_qboxes[iter, 2 * j]
                        C[1] = np_corners_qboxes[iter, 2 * j + 1]

                        D[0] = np_corners_qboxes[iter, 2 * ((j + 1) % 4)]
                        D[1] = np_corners_qboxes[iter, 2 * ((j + 1) % 4) + 1]
                        BA0 = B[0] - A[0]
                        BA1 = B[1] - A[1]
                        CA0 = C[0] - A[0]
                        CA1 = C[1] - A[1]
                        DA0 = D[0] - A[0]
                        DA1 = D[1] - A[1]
                        acd = DA1 * CA0 > CA1 * DA0
                        bcd = (D[1] - B[1]) * (C[0] - B[0]) > (C[1] - B[1]) * (D[0] - B[0])

                        if acd != bcd:
                            abc = CA1 * BA0 > BA1 * CA0
                            abd = DA1 * BA0 > BA1 * DA0
                            if abc != abd:
                                DC0 = D[0] - C[0]
                                DC1 = D[1] - C[1]
                                ABBA = A[0] * B[1] - B[0] * A[1]
                                CDDC = C[0] * D[1] - D[0] * C[1]
                                DH = BA1 * DC0 - BA0 * DC1
                                Dx = ABBA * DC0 - BA0 * CDDC
                                Dy = ABBA * DC1 - BA1 * CDDC

                                # DH = (B[1] - A[1]) * (D[0] - C[0]) - (B[0] - A[0]) * (D[1] - C[1])
                                # Dx = (A[0] * B[1] - B[0] * A[1]) * (D[0] - C[0]) - (B[0] - A[0]) * (C[0] * D[1] - D[0] * C[1])
                                det_DxA0 = B[1] * (D[0] - C[0]) + (C[0] * D[1] - D[0] * C[1])
                                det_DxA1 = - B[0] * (D[0] - C[0])
                                det_DxB0 = - A[1] * (D[0] - C[0]) - (C[0] * D[1] - D[0] * C[1])
                                det_DxB1 = A[0] * (D[0] - C[0])
                                det_DxC0 = - (A[0] * B[1] - B[0] * A[1]) - (B[0] - A[0]) * D[1]
                                det_DxC1 = (B[0] - A[0]) * D[0]
                                det_DxD0 = (A[0] * B[1] - B[0] * A[1]) + (B[0] - A[0]) * C[1]
                                det_DxD1 = -(B[0] - A[0]) * C[0]
                                # Dy = (A[0] * B[1] - B[0] * A[1]) * (D[1] - C[1]) - (B[1] - A[1]) * (C[0] * D[1] - D[0] * C[1])
                                det_DyA0 = B[1] * (D[1] - C[1])
                                det_DyA1 = - B[0] * (D[1] - C[1]) + (C[0] * D[1] - D[0] * C[1])
                                det_DyB0 = -  A[1] * (D[1] - C[1])
                                det_DyB1 = A[0] * (D[1] - C[1]) - (C[0] * D[1] - D[0] * C[1])

                                det_DyC0 = - (B[1] - A[1]) * D[1]
                                det_DyC1 = - (A[0] * B[1] - B[0] * A[1]) + (B[1] - A[1]) * D[0]
                                det_DyD0 = (B[1] - A[1]) * C[1]
                                det_DyD1 = (A[0] * B[1] - B[0] * A[1]) - (B[1] - A[1]) * C[0]
                                # DH = (B[1] - A[1]) * (D[0] - C[0]) - (B[0] - A[0]) * (D[1] - C[1])
                                det_DHA0 = (D[1] - C[1])
                                det_DHA1 = - (D[0] - C[0])
                                det_DHB0 = - (D[1] - C[1])
                                det_DHB1 = (D[0] - C[0])
                                det_DHC0 = - (B[1] - A[1])
                                det_DHC1 = (B[0] - A[0])
                                det_DHD0 = (B[1] - A[1])
                                det_DHD1 = - (B[0] - A[0])

                                DHDH = DH * DH

                                Jacbian_gboxes[iter, i * 2, n_of_inter[iter] * 2] += (
                                                                                     det_DxA0 * DH - Dx * det_DHA0) / DHDH
                                Jacbian_gboxes[iter, i * 2, n_of_inter[iter] * 2 + 1] += (
                                                                                         det_DyA0 * DH - Dy * det_DHA0) / DHDH

                                Jacbian_gboxes[iter, i * 2 + 1, n_of_inter[iter] * 2] += (
                                                                                         det_DxA1 * DH - Dx * det_DHA1) / DHDH
                                Jacbian_gboxes[iter, i * 2 + 1, n_of_inter[iter] * 2 + 1] += (
                                                                                             det_DyA1 * DH - Dy * det_DHA1) / DHDH

                                Jacbian_gboxes[iter, 2 * ((i + 1) % 4), n_of_inter[iter] * 2] += (
                                                                                                 det_DxB0 * DH - Dx * det_DHB0) / DHDH
                                Jacbian_gboxes[iter, 2 * ((i + 1) % 4), n_of_inter[iter] * 2 + 1] += (
                                                                                                     det_DyB0 * DH - Dy * det_DHB0) / DHDH

                                Jacbian_gboxes[iter, 2 * ((i + 1) % 4) + 1, n_of_inter[iter] * 2] += (
                                                                                                     det_DxB1 * DH - Dx * det_DHB1) / DHDH
                                Jacbian_gboxes[iter, 2 * ((i + 1) % 4) + 1, n_of_inter[iter] * 2 + 1] += (
                                                                                                         det_DyB1 * DH - Dy * det_DHB1) / DHDH

                                Jacbian_qboxes[iter, j * 2, n_of_inter[iter] * 2] += (
                                                                                     det_DxC0 * DH - Dx * det_DHC0) / DHDH
                                Jacbian_qboxes[iter, j * 2, n_of_inter[iter] * 2 + 1] += (
                                                                                         det_DyC0 * DH - Dy * det_DHC0) / DHDH

                                Jacbian_qboxes[iter, j * 2 + 1, n_of_inter[iter] * 2] += (
                                                                                         det_DxC1 * DH - Dx * det_DHC1) / DHDH
                                Jacbian_qboxes[iter, j * 2 + 1, n_of_inter[iter] * 2 + 1] += (
                                                                                             det_DyC1 * DH - Dy * det_DHC1) / DHDH

                                Jacbian_qboxes[iter, 2 * ((j + 1) % 4), n_of_inter[iter] * 2] += (
                                                                                                 det_DxD0 * DH - Dx * det_DHD0) / DHDH
                                Jacbian_qboxes[iter, 2 * ((j + 1) % 4), n_of_inter[iter] * 2 + 1] += (
                                                                                                     det_DyD0 * DH - Dy * det_DHD0) / DHDH

                                Jacbian_qboxes[iter, 2 * ((j + 1) % 4) + 1, n_of_inter[iter] * 2] += (
                                                                                                     det_DxD1 * DH - Dx * det_DHD1) / DHDH
                                Jacbian_qboxes[iter, 2 * ((j + 1) % 4) + 1, n_of_inter[iter] * 2 + 1] += (
                                                                                                         det_DyD1 * DH - Dy * det_DHD1) / DHDH

                                n_of_inter[iter] += 1

        tensor_Jacbian_gboxes = torch.from_numpy(Jacbian_gboxes).to(torch.device(corners_qboxes.device))
        tensor_Jacbian_qboxes = torch.from_numpy(Jacbian_qboxes).to(torch.device(corners_qboxes.device))
        grad_output_cuda = grad_output.to(torch.device(corners_qboxes.device))
        # print("grad_output_cuda =", grad_output_cuda.shape)
        tensor_grad_corners_gboxes = tensor_Jacbian_gboxes.matmul(grad_output_cuda.unsqueeze(2)).squeeze(2)
        tensor_grad_corners_qboxes = tensor_Jacbian_qboxes.matmul(grad_output_cuda.unsqueeze(2)).squeeze(2)

        return tensor_grad_corners_gboxes, tensor_grad_corners_qboxes

class sort_vertex(Function):

    @staticmethod

    def forward(ctx, int_pts, num_of_inter):

        np_int_pts       = int_pts.detach().numpy()
        np_num_of_inter  = num_of_inter.detach().numpy()
        N = int_pts.shape[0]
        np_sorted_indexs = np.zeros((N, 8), dtype=np.int32)
        sorted_int_pts   = np.zeros((N, 16), dtype=np.float32)
        for iter in range(N):
            if np_num_of_inter[iter] > 0:
                center = np.zeros((2,), dtype=np.float32)
                for i in range(np_num_of_inter[iter]):
                    center[0] += np_int_pts[iter, 2 * i]
                    center[1] += np_int_pts[iter, 2 * i + 1]
                center[0] /= np_num_of_inter[iter]
                center[1] /= np_num_of_inter[iter]

                angle = np.zeros((8,), dtype=np.float32)
                v     = np.zeros((2,), dtype=np.float32)

                for i in range(np_num_of_inter[iter]):
                    v[0] = np_int_pts[iter, 2 * i] - center[0]
                    v[1] = np_int_pts[iter, 2 * i + 1] - center[1]
                    d = math.sqrt(v[0] * v[0] + v[1] * v[1])
                    v[0] = v[0] / d
                    v[1] = v[1] / d
                    anglei = math.atan2(v[1], v[0])
                    if anglei < 0:
                        angle[i] = anglei + 2 *3.1415926
                    else:
                        angle[i] = anglei
                # sort angles with descending
                np_sorted_indexs[iter,:] = np.argsort(-angle)
                for i in range(np_num_of_inter[iter]):
                    sorted_int_pts[iter, 2 * i]     = np_int_pts[iter, 2 * np_sorted_indexs[iter, i]]
                    sorted_int_pts[iter, 2 * i + 1] = np_int_pts[iter, 2 * np_sorted_indexs[iter, i] + 1]

                    # conver numpy to tensor
        ctx.save_for_backward(int_pts, num_of_inter)
        ctx.np_sorted_indexs = np_sorted_indexs
        tensor_sorted_int_pts = torch.from_numpy(sorted_int_pts)
        return tensor_sorted_int_pts


    @staticmethod
    def backward(ctx, grad_output):

        int_pts, num_of_inter = ctx.saved_tensors
        np_sorted_indexs      = ctx.np_sorted_indexs

        N = int_pts.shape[0]
        Jacbian_int_pts = np.zeros((N, 16, 16), dtype=np.float32)
        for iter in range(N):
            for i in range(num_of_inter[iter]):
                Jacbian_int_pts[iter, 2 * np_sorted_indexs[iter, i], 2*i]         = 1
                Jacbian_int_pts[iter, 2 * np_sorted_indexs[iter, i] + 1, 2*i +1 ] = 1


        tensor_Jacbian_int_pts = torch.from_numpy(Jacbian_int_pts).to(torch.device(int_pts.device))
        grad_output_cuda = grad_output.to(torch.device(int_pts.device))
        tensor_grad_int_pts = tensor_Jacbian_int_pts.matmul(grad_output_cuda.unsqueeze(2)).squeeze(2)

        return tensor_grad_int_pts, None

class area_polygon(Function):

    @staticmethod
    def forward(ctx, int_pts, num_of_inter):
        ctx.save_for_backward(int_pts, num_of_inter)
        np_int_pts      = int_pts.detach().numpy()
        np_num_of_inter = num_of_inter.detach().numpy()
        N = int_pts.shape[0]
        areas =  np.zeros((N,),    dtype=np.float32)

        for iter in range(N):
            for i in range(np_num_of_inter[iter]-2):
                p1 = np_int_pts[iter, 0:2]
                p2 = np_int_pts[iter, 2 * i + 2:2 * i + 4]
                p3 = np_int_pts[iter, 2 * i + 4:2 * i + 6]
                areas[iter] += abs(((p1[0] - p3[0]) * (p2[1] - p3[1]) - (p1[1] - p3[1]) * (p2[0] - p3[0])) / 2.0)

        tensor_areas = torch.from_numpy(areas)

        return tensor_areas

    @staticmethod
    def backward(ctx, *grad_outputs):

        int_pts, num_of_inter = ctx.saved_tensors
        np_int_pts = int_pts.detach().numpy()
        np_num_of_inter = num_of_inter.detach().numpy()
        grad_output0 = grad_outputs[0]
        N = int_pts.shape[0]
        grad_int_pts = np.zeros((N, 16), dtype=np.float32)

        for iter in range (N):
            if(np_num_of_inter[iter] > 2):
                for i in range(np_num_of_inter[iter]):
                    if i == 0:
                        for j in range(np_num_of_inter[iter] - 2):
                            p1 = np_int_pts[iter, 0:2]
                            p2 = np_int_pts[iter, 2 * j + 2:2 * j + 4]
                            p3 = np_int_pts[iter, 2 * j + 4:2 * j + 6]

                            if((p1[0] - p3[0]) * (p2[1] - p3[1]) - (p1[1] - p3[1]) * (p2[0] - p3[0])) > 0:
                                grad_int_pts[iter, 0] += (p2[1] - p3[1])*grad_output0[iter]*0.5
                                grad_int_pts[iter, 1] += -(p2[0] - p3[0])*grad_output0[iter]*0.5
                            else:
                                grad_int_pts[iter, 0] += -(p2[1] - p3[1])*grad_output0[iter]*0.5
                                grad_int_pts[iter, 1] +=  (p2[0] - p3[0])*grad_output0[iter]*0.5

                    elif i ==1:
                        p1 = np_int_pts[iter, 0:2]
                        p2 = np_int_pts[iter, 2:4]
                        p3 = np_int_pts[iter, 4:6]
                        if ((p1[0] - p3[0]) * (p2[1] - p3[1]) - (p1[1] - p3[1]) * (p2[0] - p3[0])) > 0:
                            grad_int_pts[iter, 2]     = -(p1[1] - p3[1])*grad_output0[iter]*0.5
                            grad_int_pts[iter, 3]     = (p1[0] - p3[0])*grad_output0[iter]*0.5
                        else:
                            grad_int_pts[iter, 2]     = (p1[1] - p3[1])*grad_output0[iter]*0.5
                            grad_int_pts[iter, 3]     = -(p1[0] - p3[0])*grad_output0[iter]*0.5

                    elif i ==np_num_of_inter[iter] - 1:

                        p1 = np_int_pts[iter, 2*(np_num_of_inter[iter]-2):2*(np_num_of_inter[iter]-1)]
                        p2 = np_int_pts[iter, 2*(np_num_of_inter[iter]-1):2*(np_num_of_inter[iter])]
                        p3 = np_int_pts[iter, 0:2]
                        if ((p1[0] - p3[0]) * (p2[1] - p3[1]) - (p1[1] - p3[1]) * (p2[0] - p3[0])) > 0:
                            grad_int_pts[iter, 2*(np_num_of_inter[iter]-1)]   = - (p1[1] - p3[1])*grad_output0[iter]*0.5
                            grad_int_pts[iter, 2*np_num_of_inter[iter]-1]     = (p1[0] - p3[0])*grad_output0[iter]*0.5
                        else:
                            grad_int_pts[iter, 2*(np_num_of_inter[iter]-1)]   = (p1[1] - p3[1])*grad_output0[iter]*0.5
                            grad_int_pts[iter, 2*np_num_of_inter[iter]-1]     = - (p1[0] - p3[0])*grad_output0[iter]*0.5
                    else:
                        p1 = np_int_pts[iter, 0:2]
                        p2 = np_int_pts[iter, 2 * i - 2 : 2* i]
                        p3 = np_int_pts[iter, 2 * i  : 2 * i + 2]
                        if ((p1[0] - p3[0]) * (p2[1] - p3[1]) - (p1[1] - p3[1]) * (p2[0] - p3[0])) > 0:
                            grad_int_pts[iter, i * 2]     += (- (p2[1] - p3[1]) + (p1[1] - p3[1]))*grad_output0[iter]*0.5
                            grad_int_pts[iter, i * 2 + 1] += (- (p1[0] - p3[0]) + (p2[0] - p3[0]))*grad_output0[iter]*0.5
                        else:
                            grad_int_pts[iter, i * 2]     += ((p2[1] - p3[1]) - (p1[1] - p3[1]))*grad_output0[iter]*0.5
                            grad_int_pts[iter, i * 2 + 1] += ((p1[0] - p3[0]) - (p2[0] - p3[0]))*grad_output0[iter]*0.5

                        p1 = np_int_pts[iter, 0:2]
                        p2 = np_int_pts[iter, 2 * i : 2 * i +2]
                        p3 = np_int_pts[iter, 2 * i + 2: 2 * i + 4]
                        if ((p1[0] - p3[0]) * (p2[1] - p3[1]) - (p1[1] - p3[1]) * (p2[0] - p3[0])) > 0:
                            grad_int_pts[iter, i * 2]    += - (p1[1] - p3[1])*grad_output0[iter]*0.5
                            grad_int_pts[iter, i * 2 + 1]+= (p1[0] - p3[0])*grad_output0[iter]*0.5
                        else:
                            grad_int_pts[iter, i * 2]    +=  (p1[1] - p3[1])*grad_output0[iter]*0.5
                            grad_int_pts[iter, i * 2 + 1]+= -(p1[0] - p3[0])*grad_output0[iter]*0.5

        tensor_grad_int_pts = torch.from_numpy(grad_int_pts)
        return tensor_grad_int_pts, None
## Transform the (cx, cy, w, l, theta) representation to 4 corners representation
class rbbox_to_corners(nn.Module):

    def _init_(self, rbbox):
        super(rbbox_to_corners, self)._init_()
        self.rbbox = rbbox
        return

    def forward(ctx, rbbox):

        assert rbbox.shape[1] == 5
        device  = rbbox.device
        corners = torch.zeros((rbbox.shape[0], 8), dtype=torch.float32, device = device)
        #with torch.no_grad():
        dxcos   = rbbox[:, 2].mul(torch.cos(rbbox[:, 4])) / 2.0
        dxsin   = rbbox[:, 2].mul(torch.sin(rbbox[:, 4])) / 2.0
        dycos   = rbbox[:, 3].mul(torch.cos(rbbox[:, 4])) / 2.0
        dysin   = rbbox[:, 3].mul(torch.sin(rbbox[:, 4])) / 2.0
        corners[:, 0] = -dxcos - dysin + rbbox[:, 0]
        corners[:, 1] =  dxsin - dycos + rbbox[:, 1]
        corners[:, 2] = -dxcos + dysin + rbbox[:, 0]
        corners[:, 3] =  dxsin + dycos + rbbox[:, 1]

        corners[:, 4] =  dxcos + dysin + rbbox[:, 0]
        corners[:, 5] = -dxsin + dycos + rbbox[:, 1]
        corners[:, 6] =  dxcos - dysin + rbbox[:, 0]
        corners[:, 7] = -dxsin - dycos + rbbox[:, 1]
            # generate clockwise corners and rotate it clockwise
        #ctx.save_for_backward(rbbox)
        return corners



'''
## Transform the (cx, cy, w, l, theta) representation to 4 corners representation
class rbbox_to_corners(Function):
    @staticmethod
    def forward(ctx, rbbox):
        #with torch.no_grad():
        #N = rbbox.shape[0]
        #center_x = rbbox[:, 0]
        #center_y = rbbox[:, 1]
        #x_d      = rbbox[:, 2]
        #y_d      = rbbox[:, 3]
        #angle    = rbbox[:, 4]
        assert rbbox.shape[1] == 5

        device  = rbbox.device
        corners = torch.zeros((rbbox.shape[0], 8), dtype=torch.float32, device = device)
        with torch.no_grad():
            dxcos   = rbbox[:, 2].mul(torch.cos(rbbox[:, 4])) / 2.0
            dxsin   = rbbox[:, 2].mul(torch.sin(rbbox[:, 4])) / 2.0
            dycos   = rbbox[:, 3].mul(torch.cos(rbbox[:, 4])) / 2.0
            dysin   = rbbox[:, 3].mul(torch.sin(rbbox[:, 4])) / 2.0
            corners[:, 0] = -dxcos + dysin + rbbox[:, 0]
            corners[:, 1] = -dxsin - dycos + rbbox[:, 1]
            corners[:, 2] = -dxcos - dysin + rbbox[:, 0]
            corners[:, 3] = -dxsin + dycos + rbbox[:, 1]
            corners[:, 4] =  dxcos - dysin + rbbox[:, 0]
            corners[:, 5] =  dxsin + dycos + rbbox[:, 1]
            corners[:, 6] =  dxcos + dysin + rbbox[:, 0]
            corners[:, 7] =  dxsin - dycos + rbbox[:, 1]
        # generate clockwise corners and rotate it clockwise
        ctx.save_for_backward(rbbox)

        return corners
    @staticmethod
    def backward(ctx, *grad_outputs):
        variables = ctx.saved_variables

        rbbox = variables[0]
        grad_output0 = grad_outputs[0]
        #center_x   = rbbox[:, 0]
        #center_y   = rbbox[:, 1]
        #x_d        = rbbox[:, 2]
        #y_d        = rbbox[:, 3]
        #angle      = rbbox[:, 4]
        N = rbbox.shape[0]
        device     = rbbox.device
        grad_rbbox = torch.zeros((N, 5), dtype=torch.float32, device = device)

        cos_theta  = torch.cos(rbbox[:, 4])
        sin_theta  = torch.sin(rbbox[:, 4])
        d_dxcos = -rbbox[:, 2].mul(torch.sin(rbbox[:, 4])) / 2.0
        d_dxsin =  rbbox[:, 2].mul(torch.cos(rbbox[:, 4])) / 2.0
        d_dycos = -rbbox[:, 3].mul(torch.sin(rbbox[:, 4])) / 2.0
        d_dysin =  rbbox[:, 3].mul(torch.cos(rbbox[:, 4])) / 2.0
        # for center_x
        grad_rbbox[:, 0] = (grad_output0[:, 0]  + grad_output0[:, 2] + grad_output0[:, 4] + grad_output0[:, 6])
        # for center_y
        grad_rbbox[:, 1] = (grad_output0[:, 1]  + grad_output0[:, 3] + grad_output0[:, 5] + grad_output0[:, 7])
        # for x_d
        grad_rbbox[:, 2] =(- grad_output0[:,0].mul(cos_theta) - grad_output0[:,1].mul(sin_theta)
                           - grad_output0[:,2].mul(cos_theta) - grad_output0[:,3].mul(sin_theta)
                           + grad_output0[:,4].mul(cos_theta) + grad_output0[:,5].mul(sin_theta)
                           + grad_output0[:,6].mul(cos_theta) + grad_output0[:,7].mul(sin_theta)) / 2.0
        # for grad_outputs
        grad_rbbox[:, 3] =( grad_output0[:,0].mul(sin_theta) - grad_output0[:,1].mul(cos_theta)
                          - grad_output0[:,2].mul(sin_theta) + grad_output0[:,3].mul(cos_theta)
                          - grad_output0[:,4].mul(sin_theta) + grad_output0[:,5].mul(cos_theta)
                          + grad_output0[:,6].mul(sin_theta) - grad_output0[:,7].mul( cos_theta)) / 2.0
        # for theta
        grad_rbbox[:, 4] = ( grad_output0[:,0].mul(-d_dxcos + d_dysin)
                           + grad_output0[:,1].mul(-d_dxsin - d_dycos)
                           + grad_output0[:,2].mul(-d_dxcos - d_dysin)
                           + grad_output0[:,3].mul(-d_dxsin + d_dycos)
                           + grad_output0[:,4].mul(d_dxcos - d_dysin)
                           + grad_output0[:,5].mul(d_dxsin + d_dycos)
                           + grad_output0[:,6].mul(d_dxcos + d_dysin)
                           + grad_output0[:,7].mul(d_dxsin - d_dycos))

        return grad_rbbox
'''
class rinter_area_compute(nn.Module):

    def _init_(self, corners_gboxes, corners_qboxes):
        super(rinter_area_compute, self)._init_()
        self.corners_gboxes = corners_gboxes
        self.corners_qboxes = corners_qboxes
        return
    # Compute the inter area of two rotated rectangles

    def forward(ctx, corners_gboxes, corners_qboxes):
        intersections, num_of_intersections = compute_vertex(corners_gboxes, corners_qboxes)
        sorted_int_pts = sort_vertex(intersections, num_of_intersections)
        inter_area     = area_polygon(sorted_int_pts, num_of_intersections)
        return inter_area

## Find the convex hull from a set of point cloud
class find_convex_hull(Function):
    # get the minimum bounding box from a set of points
    @staticmethod
    def forward(ctx, corners):
        np_corners = corners.cpu().detach().numpy()
        hull = ConvexHull(np_corners)
        M = hull.nsimplex
        index = hull.vertices
        hull_points_2d = np.zeros((M, 2), np.float32)
        for i in range(M):
            hull_points_2d[i, 0] = np_corners[index[i], 0]
            hull_points_2d[i, 1] = np_corners[index[i], 1]
        tensor_hull_points_2d = torch.from_numpy(hull_points_2d).to(torch.device(corners.device))
        ctx.index = index
        return tensor_hull_points_2d

    @staticmethod
    def backward(ctx, *grad_outputs):
        index        = ctx.index
        grad_output0 = grad_outputs[0]
        device       = grad_output0.device
        grad_corners = torch.zeros((8, 2), dtype=torch.float32, device=device)
        for i in range(len(index)):
            grad_corners[index[i], 0] = grad_output0[i, 0]
            grad_corners[index[i], 1] = grad_output0[i, 1]
        return grad_corners

## nn Module
class mbr_convex_hull(nn.Module):

    def _init_(self, hull_points_2d):
        super(mbr_convex_hull, self)._init_()
        self.hull_points_2d = hull_points_2d
        return

    def forward(ctx, hull_points_2d):
        device = hull_points_2d.device
        N = hull_points_2d.shape[0]
        edges = hull_points_2d[1:N, :].add(- hull_points_2d[0:N - 1, :])
        edge_angles = torch.atan2(edges[:, 1], edges[:, 0])
        edge_angles = torch.fmod(edge_angles, 3.1415926/2.0)
        edge_angles = torch.abs(edge_angles)
        #edge_angles = torch.unique(edge_angles)
        #print("edge_angles =", edge_angles)
        a = torch.stack((torch.cos(edge_angles), torch.cos(edge_angles - 3.1415926 / 2.0)), 1)
        a = torch.unsqueeze(a, 1)
        b = torch.stack((torch.cos(edge_angles + 3.1415926 / 2.0), torch.cos(edge_angles)),  1)
        b = torch.unsqueeze(b, 1)
        R_tensor   = torch.cat((a, b), 1)
        hull_points_2d_ = torch.unsqueeze(torch.transpose(hull_points_2d, 0, 1),0)
        rot_points = R_tensor.matmul(hull_points_2d_)
        min_x  = torch.min(rot_points, 2)[0]
        max_x  = torch.max(rot_points, 2)[0]
        areas = (max_x[:,0]- min_x[:,0]).mul(max_x[:,1] - min_x[:,1])
        return torch.min(areas)


'''
class mbr_area_compute(nn.Module):
    # get the minimum bounding box from a set of points

    def _init_(self, corners_gboxes, corners_qboxes):
        super(mbr_area_compute, self)._init_()
        self.corners_gboxes = corners_gboxes
        self.corners_qboxes = corners_qboxes
        return

    def forward(ctx, corners_gboxes, corners_qboxes):
        # np_corners = corners.numpy()
        N = corners_gboxes.shape[0]
        mbr_rect_areas   = torch.zeros((N,), dtype=torch.float32)
        #mbr_rect_areas   = torch.zeros((N,), dtype=torch.float32, device = corners_gboxes.device)
        corners_gboxes_1 = torch.stack((corners_gboxes[:, [0, 2, 4, 6]], corners_gboxes[:, [1, 3, 5, 7]]), 2)
        corners_qboxes_1 = torch.stack((corners_qboxes[:, [0, 2, 4, 6]], corners_qboxes[:, [1, 3, 5, 7]]), 2)
        corners_pts      = torch.cat((corners_gboxes_1, corners_qboxes_1), 1)
        for iter in range(N):
            convex_hull_pts        = find_convex_hull(corners_pts[iter,:,:].squeeze())
            mbr_convex_hull_object = mbr_convex_hull()
            mbr_rect_areas[iter]   = mbr_convex_hull_object(convex_hull_pts)
        return mbr_rect_areas
'''
class mbr_area_compute(nn.Module):
    # get the minimum bounding box from a set of points

    def _init_(self, corners):
        super(mbr_area_compute, self)._init_()
        self.corners = corners
        return

    def forward(ctx, corners):
        # np_corners = corners.numpy()
        N = corners.shape[0]
        # mbr_rect_areas   = torch.zeros((N,), dtype=torch.float32)
        mbr_rect_area = []
        for i in range(N):
            mbr_rect_area.append(torch.zeros((1,), dtype=torch.float32, device=corners.device))
        # mbr_rect_areas   = torch.zeros((N,), dtype=torch.float32, device = corners_gboxes.device)
        for iter in range(N):
            convex_hull_pts        = find_convex_hull(corners[iter, :, :].squeeze())
            mbr_convex_hull_object = mbr_convex_hull()
            mbr_rect_area[iter]    = mbr_convex_hull_object(convex_hull_pts)
        mbr_rect_areas             = torch.stack(mbr_rect_area)  # torch.cat(mbr_rect_area)
        # ctx.save_for_backward(corners)
        return mbr_rect_areas
##

class _second_box_decode_operation(nn.Module):
    """box decode for VoxelNet in lidar
    Args:
        boxes ([N, 7] Tensor): normal boxes: x, y, z, w, l, h, r
        anchors ([N, 7] Tensor): anchors
    """
    # need to convert box_encodings to z-bottom format

    def _init_(self, box_encodings,anchors,encode_angle_to_vector, smooth_dim):
        super(_second_box_decode_operation, self)._init_()
        self.box_encodings = box_encodings
        self.anchors = anchors
        self.encode_angle_to_vector = False
        self.smooth_dim = False
        return

    def forward(ctx, box_encodings,anchors,encode_angle_to_vector, smooth_dim):

        """box decode for VoxelNet in lidar
        Args:
            boxes ([N, 7] Tensor): normal boxes: x, y, z, w, l, h, r
            anchors ([N, 7] Tensor): anchors
        """
        xa, ya, za, wa, la, ha, ra = torch.split(anchors, 1, dim=-1)
        if encode_angle_to_vector:
            xt, yt, zt, wt, lt, ht, rtx, rty = torch.split(box_encodings, 1, dim=-1)
        else:
            xt, yt, zt, wt, lt, ht, rt = torch.split(box_encodings, 1, dim=-1)
        # xt, yt, zt, wt, lt, ht, rt = torch.split(box_encodings, 1, dim=-1)
        za = za + ha / 2
        diagonal = torch.sqrt(la ** 2 + wa ** 2)
        xg = xt * diagonal + xa
        yg = yt * diagonal + ya
        zg = zt * ha + za
        if smooth_dim:
            lg = (lt + 1) * la
            wg = (wt + 1) * wa
            hg = (ht + 1) * ha
        else:

            lg = torch.exp(lt) * la
            wg = torch.exp(wt) * wa
            hg = torch.exp(ht) * ha
        if encode_angle_to_vector:
            rax = torch.cos(ra)
            ray = torch.sin(ra)
            rgx = rtx + rax
            rgy = rty + ray
            rg = torch.atan2(rgy, rgx)
        else:
            rg = rt + ra
        zg = zg - hg / 2
        return torch.cat([xg, yg, zg, wg, lg, hg, rg], dim=-1)

###################################
#simplified version
###################################

class rbbox_corners_aligned(nn.Module):


    def _init_(self, gboxes):
        super(rbbox_corners_aligned, self)._init_()
        self.corners_gboxes = gboxes
        return

    def forward(ctx, gboxes):
        # generate clockwise corners and rotate it clockwise
        eps = 0.0
        N = gboxes.shape[0]
        center_x = gboxes[:, 0]
        center_y = gboxes[:, 1]
        x_d = gboxes[:, 2]
        y_d = gboxes[:, 3]
        corners = torch.zeros([N, 2, 4], device= gboxes.device, dtype=torch.float32)
        corners[:, 0, 0] = x_d.mul(-0.5)
        corners[:, 1, 0] = y_d.mul(-0.5)

        corners[:, 0, 1] = x_d.mul(-0.5)
        corners[:, 1, 1] = y_d.mul(0.5)

        corners[:, 0, 2] = x_d.mul(0.5)
        corners[:, 1, 2] = y_d.mul(0.5)

        corners[:, 0, 3] = x_d.mul(0.5)
        corners[:, 1, 3] = y_d.mul(-0.5)

        b = center_x.unsqueeze(1).repeat(1, 4).unsqueeze(1)
        c = center_y.unsqueeze(1).repeat(1, 4).unsqueeze(1)

        return (corners + torch.cat((b, c), 1))

class align_inter_aligned(nn.Module):

    def _init_(self, gboxes,qboxes):
        super(align_inter_aligned, self)._init_()
        self.gboxes = gboxes
        self.qboxes = qboxes
        return
    def forward(ctx, gboxes, qboxes):
        N = gboxes.shape[0]
        M = qboxes.shape[0]
        eps = 0.0000000000000001
        assert N == M

        ## we can project the 3D bounding boxes into 3 different plane
        ## view1 xoz plane
        inter_area_xoz = torch.zeros((N,), device=gboxes.device, dtype=torch.float32)
        mbr_area_xoz = torch.zeros((N,), device=gboxes.device, dtype=torch.float32)
        rbbox_corners_aligned_object = rbbox_corners_aligned()
        rotated_corners1 = rbbox_corners_aligned_object(gboxes[:, [0, 2, 3, 5, 6]])
        rotated_corners2 = rbbox_corners_aligned_object(qboxes[:, [0, 2, 3, 5, 6]])
        for i in range(N):
            iw = (min(rotated_corners1[i, 0, 1], rotated_corners2[i, 0, 3]) -
                  max(rotated_corners1[i, 0, 0], rotated_corners2[i, 0, 3]) + eps)
            if (iw > 0):
                ih = ((min(rotated_corners1[i, 1, 1], rotated_corners2[i, 1, 1]) -
                       max(rotated_corners1[i, 1, 0], rotated_corners2[i, 1, 0]) + eps))
                if (ih > 0):
                    inter_area_xoz[i] = iw * ih

            iwmbr = (max(rotated_corners1[i, 0, 3], rotated_corners2[i, 0, 3]) -
                     min(rotated_corners1[i, 0, 0], rotated_corners2[i, 0, 0]) + eps)

            ihmbr = ((max(rotated_corners1[i, 1, 1], rotated_corners2[i, 1, 1]) -
                      min(rotated_corners1[i, 1, 0], rotated_corners2[i, 1, 0]) + eps))
            mbr_area_xoz[i] = iwmbr*ihmbr

        ### view2 xoy plane
        inter_area_xoy = torch.zeros((N,), device=gboxes.device, dtype=torch.float32)
        mbr_area_xoy = torch.zeros((N,), device=gboxes.device, dtype=torch.float32)
        rotated_corners1 = rbbox_corners_aligned_object(gboxes[:, [0, 1, 3, 4, 6]])
        rotated_corners2 = rbbox_corners_aligned_object(qboxes[:, [0, 1, 3, 4, 6]])
        for i in range(N):
            iw = (min(rotated_corners1[i, 0, 1], rotated_corners2[i, 0, 3]) -
                  max(rotated_corners1[i, 0, 0], rotated_corners2[i, 0, 3]) + eps)
            if (iw > 0):
                ih = ((min(rotated_corners1[i, 1, 1], rotated_corners2[i, 1, 1]) -
                       max(rotated_corners1[i, 1, 0], rotated_corners2[i, 1, 0]) + eps))
                if (ih > 0):
                    inter_area_xoy[i] = iw * ih

            iwmbr = (max(rotated_corners1[i, 0, 3], rotated_corners2[i, 0, 3]) -
                     min(rotated_corners1[i, 0, 0], rotated_corners2[i, 0, 0]) + eps)

            ihmbr = ((max(rotated_corners1[i, 1, 1], rotated_corners2[i, 1, 1]) -
                      min(rotated_corners1[i, 1, 0], rotated_corners2[i, 1, 0]) + eps))

            mbr_area_xoy[i] = iwmbr * ihmbr

        ### view3 yoz plane
        inter_area_yoz = torch.zeros((N,), device=gboxes.device, dtype=torch.float32)
        mbr_area_yoz = torch.zeros((N,), device=gboxes.device, dtype=torch.float32)
        rotated_corners1 = rbbox_corners_aligned_object(gboxes[:, [1, 2, 4, 5, 6]])
        rotated_corners2 = rbbox_corners_aligned_object(qboxes[:, [1, 2, 4, 5, 6]])
        for i in range(N):
            iw = (min(rotated_corners1[i, 0, 1], rotated_corners2[i, 0, 3]) -
                  max(rotated_corners1[i, 0, 0], rotated_corners2[i, 0, 3]) + eps)
            if (iw > 0):
                ih = ((min(rotated_corners1[i, 1, 1], rotated_corners2[i, 1, 1]) -
                       max(rotated_corners1[i, 1, 0], rotated_corners2[i, 1, 0]) + eps))
                if (ih > 0):
                    inter_area_yoz[i] = iw * ih

            iwmbr = (max(rotated_corners1[i, 0, 3], rotated_corners2[i, 0, 3]) -
                     min(rotated_corners1[i, 0, 0], rotated_corners2[i, 0, 0]) + eps)

            ihmbr = ((max(rotated_corners1[i, 1, 1], rotated_corners2[i, 1, 1]) -
                      min(rotated_corners1[i, 1, 0], rotated_corners2[i, 1, 0]) + eps))
            mbr_area_yoz[i] = iwmbr * ihmbr

        return inter_area_xoz, mbr_area_xoz, inter_area_xoy, mbr_area_xoy, inter_area_yoz, mbr_area_yoz

###################################
#simplified version
###################################

class gious_2D(nn.Module):
    # Compute the inter area of two rotated rectangles
    def _init_(self, gboxes, qboxes):
        super(gious_2D, self)._init_()
        self.gboxes = gboxes
        self.qboxes = qboxes
        return

    # Compute the inter area of two rotated rectangles
    def forward(ctx, gboxes, qboxes):
        # get the corners of the rectangles

        corners_gboxes = rbbox_to_corners(gboxes[:, [0, 1, 3, 4, 6]])
        corners_qboxes = rbbox_to_corners(qboxes[:, [0, 1, 3, 4, 6]])
        # compute the inter area
        rinter_area_compute_object = rinter_area_compute()
        inter_area = rinter_area_compute_object(corners_gboxes, corners_qboxes)
        # compute the mbr area
        mbr_area_compute_object = mbr_area_compute()
        mbr_area   = mbr_area_compute_object(corners_gboxes, corners_qboxes)

        union_area = gboxes[:,2].mul(gboxes[:,3]) + qboxes[:,2].mul(qboxes[:,3]) - inter_area
        ious  = torch.div(inter_area, union_area)
        gious = ious - torch.div(mbr_area - union_area, mbr_area)
        return gious

class gious_3D(nn.Module):
    # Compute the inter area of two rotated rectangles
    def _init_(self, gboxes, qboxes, aligned=False):
        super(gious_3D, self)._init_()
        self.gboxes = gboxes
        self.qboxes = qboxes
        self.aligned = aligned
        return

    def forward(ctx, gboxes, qboxes, aligned):
        assert gboxes.shape[0] == qboxes.shape[0]
        indicator = torch.gt(gboxes[:, 3], 0) & torch.gt(gboxes[:, 4], 0) & torch.gt(gboxes[:, 5], 0) \
                    & torch.gt(qboxes[:, 3], 0) & torch.gt(qboxes[:, 4], 0) & torch.gt(qboxes[:, 5], 0)
        index_loc = torch.nonzero(indicator)
        ## if we want to compute the gious of two aligned rectangles
        gious = torch.zeros([gboxes.shape[0], ], device=gboxes.device, dtype=torch.float32)

        if (aligned):

            align_inter_aligned_object = align_inter_aligned()
            inter_area_xoz, mbr_area_xoz, inter_area_xoy, mbr_area_xoy, inter_area_yoz, mbr_area_yoz = align_inter_aligned_object(gboxes, qboxes)
            volume_gboxes = gboxes[:, 3].mul(gboxes[:, 4]).mul(gboxes[:, 5])
            volume_qboxes = qboxes[:, 3].mul(qboxes[:, 4]).mul(qboxes[:, 5])
            ## for three different views xoz plane
            #inter_h = (torch.min(gboxes[:, 1], qboxes[:, 1]) - torch.max(gboxes[:, 1] - gboxes[:, 4], qboxes[:, 1] - qboxes[:, 4]))
            #oniou_h = (torch.max(gboxes[:, 1], qboxes[:, 1]) - torch.min(gboxes[:, 1] - gboxes[:, 4], qboxes[:, 1] - qboxes[:, 4]))
            #inter_h[inter_h < 0] = 0
            #oniou_h[oniou_h < 0] = 0
            #inter_area_xoz_cuda = inter_area_xoz.to(torch.device(gboxes.device))
            #mbr_area_xoz_cuda   = mbr_area_xoz.to(torch.device(gboxes.device))
            #volume_inc = inter_h.mul(inter_area_xoz_cuda)
            #volume_con = oniou_h.mul(mbr_area_xoz_cuda)
            #volume_union = (volume_gboxes + volume_qboxes - volume_inc)
            #volume_ca = volume_con - volume_union
            #ious = torch.div(volume_inc, volume_union)
            union_xoz = gboxes[:, 3].mul(gboxes[:, 5]) + qboxes[:, 3].mul(qboxes[:, 5]) - inter_area_xoz
            iou_xoz   = torch.div(inter_area_xoz, union_xoz)
            iou_bis_xoz = torch.div(mbr_area_xoz - union_xoz,mbr_area_xoz)
            gious_xoz = iou_xoz - iou_bis_xoz
            ## for xoy plane
            union_xoy = gboxes[:, 3].mul(gboxes[:, 4]) + qboxes[:, 3].mul(qboxes[:, 4]) - inter_area_xoy
            iou_xoy = torch.div(inter_area_xoy, union_xoy)
            iou_bis_xoy = torch.div(mbr_area_xoy - union_xoy, mbr_area_xoy)
            gious_xoy = iou_xoy - iou_bis_xoy
            ## for yoz plane
            union_yoz = gboxes[:, 4].mul(gboxes[:, 5]) + qboxes[:, 4].mul(qboxes[:, 5]) - inter_area_xoy
            iou_yoz = torch.div(inter_area_yoz, union_yoz)
            iou_bis_yoz = torch.div(mbr_area_yoz - union_yoz, mbr_area_yoz)
            gious_xoy = iou_yoz - iou_bis_yoz
            gious[index_loc[:, 0]] = (gious_xoz[index_loc[:, 0]] + gious_xoy[index_loc[:, 0]]  + gious_xoy[index_loc[:, 0]])/3.0

            #for i in range(inter_area_xoz.shape[0]):
            #    if (gious[i] > 1):
            #        print("infor: (%.4f %.4f %.4f %.4f %.4f %.4f %.4f,%.4f %.4f %.4f %.4f)"
            #              % (i, inter_h[i], oniou_h[i], inter_area_xoz[i], mbr_area_xoz[i], ious[i], gious[i], volume_inc[i],
            #                 volume_con[i], volume_union[i], volume_ca[i]))
            #    elif (gious[i] < -1):
            #        print("infor: (%.4f %.4f %.4f %.4f %.4f %.4f %.4f,%.4f %.4f %.4f %.4f)"
            #              % (i, inter_h[i], oniou_h[i], inter_area_xoz[i], mbr_area_xoz[i], ious[i], gious[i], volume_inc[i],
            #                 volume_con[i], volume_union[i], volume_ca[i]))
        else:
            rbbox_to_corners_object = rbbox_to_corners()
            corners_gboxes = rbbox_to_corners_object(gboxes[:, [0, 2, 3, 5, 6]])
            corners_qboxes = rbbox_to_corners_object(qboxes[:, [0, 2, 3, 5, 6]])
            # compute the inter area
            rinter_area_compute_object = rinter_area_compute()
            inter_area = rinter_area_compute_object(corners_gboxes, corners_qboxes)

            corners_gboxes_1 = torch.stack((corners_gboxes[:, [0, 2, 4, 6]], corners_gboxes[:, [1, 3, 5, 7]]), 2)
            corners_qboxes_1 = torch.stack((corners_qboxes[:, [0, 2, 4, 6]], corners_qboxes[:, [1, 3, 5, 7]]), 2)
            corners_pts = torch.cat((corners_gboxes_1, corners_qboxes_1), 1)

            # compute the mbr area
            mbr_area_compute_object = mbr_area_compute()
            mbr_area = mbr_area_compute_object(corners_pts)

            ## Compute the gious for 3D
            inter_h = (torch.min(gboxes[:, 1], qboxes[:, 1]) - torch.max(gboxes[:, 1] - gboxes[:, 4], qboxes[:, 1] - qboxes[:, 4]))
            oniou_h = (torch.max(gboxes[:, 1], qboxes[:, 1]) - torch.min(gboxes[:, 1] - gboxes[:, 4], qboxes[:, 1] - qboxes[:, 4]))
            inter_h[inter_h < 0] = 0
            volume_gboxes = gboxes[:, 3].mul(gboxes[:, 4]).mul(gboxes[:, 5])
            volume_qboxes = qboxes[:, 3].mul(qboxes[:, 4]).mul(qboxes[:, 5])
            inter_area_cuda = inter_area.to(torch.device(gboxes.device))
            mbr_area_cuda   = mbr_area.to(torch.device(gboxes.device))
            volume_inc = inter_h.mul(inter_area_cuda)
            volume_con = oniou_h.mul(mbr_area_cuda)
            volume_union = (volume_gboxes + volume_qboxes - volume_inc)
            volume_ca    = volume_con - volume_union
            ious         = torch.div(volume_inc, volume_union)

            gious = torch.zeros([gboxes.shape[0],], device=gboxes.device, dtype=torch.float32)
            gious[index_loc[:, 0]] = ious[index_loc[:, 0]] - torch.div(volume_ca[index_loc[:, 0]], volume_con[index_loc[:, 0]])
            for i in range(inter_area.shape[0]):
                if(gious[i] < -1):
                    print("infor: (%.4f %.4f %.4f %.4f %.4f %.4f %.4f,%.4f %.4f %.4f %.4f)"
                      %(i,inter_h[i], oniou_h[i], inter_area[i], mbr_area[i], ious[i], gious[i],volume_inc[i],volume_con[i],volume_union[i],volume_ca[i]))

        return torch.unsqueeze(gious,1)

class ious_3D(nn.Module):
    # Compute the inter area of two rotated rectangles
    def _init_(self, gboxes, qboxes, aligned=False):
        super(ious_3D, self)._init_()
        self.gboxes  = gboxes
        self.qboxes  = qboxes
        self.aligned = aligned
        return

    def forward(ctx, gboxes, qboxes, aligned=False):
        assert gboxes.shape[0] == qboxes.shape[0]
        indicator = torch.gt(gboxes[:, 3], 0) & torch.gt(gboxes[:, 4], 0) & torch.gt(gboxes[:, 5], 0) \
                    & torch.gt(qboxes[:, 3], 0) & torch.gt(qboxes[:, 4], 0) & torch.gt(qboxes[:, 5], 0)
        index_loc = torch.nonzero(indicator)
        ## if we want to compute the gious of two aligned rectangles
        gious = torch.zeros([gboxes.shape[0], ], device=gboxes.device, dtype=torch.float32)

        if (aligned):

            align_inter_aligned_object = align_inter_aligned()
            inter_area_xoz, mbr_area_xoz, inter_area_xoy, mbr_area_xoy, inter_area_yoz, mbr_area_yoz = align_inter_aligned_object(gboxes, qboxes)
            volume_gboxes = gboxes[:, 3].mul(gboxes[:, 4]).mul(gboxes[:, 5])
            volume_qboxes = qboxes[:, 3].mul(qboxes[:, 4]).mul(qboxes[:, 5])

            union_xoz = gboxes[:, 3].mul(gboxes[:, 5]) + qboxes[:, 3].mul(qboxes[:, 5]) - inter_area_xoz
            iou_xoz   = torch.div(inter_area_xoz, union_xoz)
            iou_bis_xoz = torch.div(mbr_area_xoz - union_xoz,mbr_area_xoz)
            gious_xoz = iou_xoz - iou_bis_xoz
            ## for xoy plane
            union_xoy = gboxes[:, 3].mul(gboxes[:, 4]) + qboxes[:, 3].mul(qboxes[:, 4]) - inter_area_xoy
            iou_xoy = torch.div(inter_area_xoy, union_xoy)
            iou_bis_xoy = torch.div(mbr_area_xoy - union_xoy, mbr_area_xoy)
            gious_xoy = iou_xoy - iou_bis_xoy
            ## for yoz plane
            union_yoz = gboxes[:, 4].mul(gboxes[:, 5]) + qboxes[:, 4].mul(qboxes[:, 5]) - inter_area_xoy
            iou_yoz = torch.div(inter_area_yoz, union_yoz)
            iou_bis_yoz = torch.div(mbr_area_yoz - union_yoz, mbr_area_yoz)
            gious_xoy = iou_yoz - iou_bis_yoz
            gious[index_loc[:, 0]] = (gious_xoz[index_loc[:, 0]] + gious_xoy[index_loc[:, 0]]  + gious_xoy[index_loc[:, 0]])/3.0

        else:
            rbbox_to_corners_object = rbbox_to_corners()
            corners_gboxes = rbbox_to_corners_object(gboxes[:, [0, 2, 3, 5, 6]])
            corners_qboxes = rbbox_to_corners_object(qboxes[:, [0, 2, 3, 5, 6]])

            rinter_area_compute_object = rinter_area_compute()
            inter_area = rinter_area_compute_object(corners_gboxes, corners_qboxes)

            inter_h = (torch.min(gboxes[:, 1], qboxes[:, 1]) - torch.max(gboxes[:, 1] - gboxes[:, 4], qboxes[:, 1] - qboxes[:, 4]))
            #oniou_h = (torch.max(gboxes[:, 1], qboxes[:, 1]) - torch.min(gboxes[:, 1] - gboxes[:, 4], qboxes[:, 1] - qboxes[:, 4]))
            inter_h[inter_h < 0] = 0
            volume_gboxes = gboxes[:, 3].mul(gboxes[:, 4]).mul(gboxes[:, 5])
            volume_qboxes = qboxes[:, 3].mul(qboxes[:, 4]).mul(qboxes[:, 5])
            inter_area_cuda = inter_area.to(torch.device(gboxes.device))

            volume_inc = inter_h.mul(inter_area_cuda)
            volume_union = (volume_gboxes + volume_qboxes - volume_inc)
            ious = torch.zeros([gboxes.shape[0],], device=gboxes.device, dtype=torch.float32)
            ious[index_loc[:, 0]] = torch.div(volume_inc[index_loc[:, 0]], volume_union[index_loc[:, 0]])

        return torch.unsqueeze(ious,1)

    class gious_3D(nn.Module):
        # Compute the inter area of two rotated rectangles
        def _init_(self, gboxes, qboxes, aligned=False):
            super(gious_3D, self)._init_()
            self.gboxes = gboxes
            self.qboxes = qboxes
            self.aligned = aligned
            return

        def forward(ctx, gboxes, qboxes, aligned):

            assert gboxes.shape[0] == qboxes.shape[0]

            indicator = torch.gt(gboxes[:, 3], 0) & torch.gt(gboxes[:, 4], 0) & torch.gt(gboxes[:, 5], 0) \
                        & torch.gt(qboxes[:, 3], 0) & torch.gt(qboxes[:, 4], 0) & torch.gt(qboxes[:, 5], 0)
            index_loc = torch.nonzero(indicator)
            ## if we want to compute the gious of two aligned rectangles
            gious = torch.zeros([gboxes.shape[0], ], device=gboxes.device, dtype=torch.float32)

            if (aligned):

                align_inter_aligned_object = align_inter_aligned()
                inter_area_xoz, mbr_area_xoz, inter_area_xoy, mbr_area_xoy, inter_area_yoz, mbr_area_yoz = align_inter_aligned_object(
                    gboxes, qboxes)
                volume_gboxes = gboxes[:, 3].mul(gboxes[:, 4]).mul(gboxes[:, 5])
                volume_qboxes = qboxes[:, 3].mul(qboxes[:, 4]).mul(qboxes[:, 5])
                ## for three different views xoz plane
                # inter_h = (torch.min(gboxes[:, 1], qboxes[:, 1]) - torch.max(gboxes[:, 1] - gboxes[:, 4], qboxes[:, 1] - qboxes[:, 4]))
                # oniou_h = (torch.max(gboxes[:, 1], qboxes[:, 1]) - torch.min(gboxes[:, 1] - gboxes[:, 4], qboxes[:, 1] - qboxes[:, 4]))
                # inter_h[inter_h < 0] = 0
                # oniou_h[oniou_h < 0] = 0
                # inter_area_xoz_cuda = inter_area_xoz.to(torch.device(gboxes.device))
                # mbr_area_xoz_cuda   = mbr_area_xoz.to(torch.device(gboxes.device))
                # volume_inc = inter_h.mul(inter_area_xoz_cuda)
                # volume_con = oniou_h.mul(mbr_area_xoz_cuda)
                # volume_union = (volume_gboxes + volume_qboxes - volume_inc)
                # volume_ca = volume_con - volume_union
                # ious = torch.div(volume_inc, volume_union)
                union_xoz = gboxes[:, 3].mul(gboxes[:, 5]) + qboxes[:, 3].mul(qboxes[:, 5]) - inter_area_xoz
                iou_xoz = torch.div(inter_area_xoz, union_xoz)
                iou_bis_xoz = torch.div(mbr_area_xoz - union_xoz, mbr_area_xoz)
                gious_xoz = iou_xoz - iou_bis_xoz
                ## for xoy plane
                union_xoy = gboxes[:, 3].mul(gboxes[:, 4]) + qboxes[:, 3].mul(qboxes[:, 4]) - inter_area_xoy
                iou_xoy = torch.div(inter_area_xoy, union_xoy)
                iou_bis_xoy = torch.div(mbr_area_xoy - union_xoy, mbr_area_xoy)
                gious_xoy = iou_xoy - iou_bis_xoy
                ## for yoz plane
                union_yoz = gboxes[:, 4].mul(gboxes[:, 5]) + qboxes[:, 4].mul(qboxes[:, 5]) - inter_area_xoy
                iou_yoz = torch.div(inter_area_yoz, union_yoz)
                iou_bis_yoz = torch.div(mbr_area_yoz - union_yoz, mbr_area_yoz)
                gious_xoy = iou_yoz - iou_bis_yoz
                gious[index_loc[:, 0]] = (gious_xoz[index_loc[:, 0]] + gious_xoy[index_loc[:, 0]] + gious_xoy[
                    index_loc[:, 0]]) / 3.0

                # for i in range(inter_area_xoz.shape[0]):
                #    if (gious[i] > 1):
                #        print("infor: (%.4f %.4f %.4f %.4f %.4f %.4f %.4f,%.4f %.4f %.4f %.4f)"
                #              % (i, inter_h[i], oniou_h[i], inter_area_xoz[i], mbr_area_xoz[i], ious[i], gious[i], volume_inc[i],
                #                 volume_con[i], volume_union[i], volume_ca[i]))
                #    elif (gious[i] < -1):
                #        print("infor: (%.4f %.4f %.4f %.4f %.4f %.4f %.4f,%.4f %.4f %.4f %.4f)"
                #              % (i, inter_h[i], oniou_h[i], inter_area_xoz[i], mbr_area_xoz[i], ious[i], gious[i], volume_inc[i],
                #                 volume_con[i], volume_union[i], volume_ca[i]))
            else:
                corners_gboxes = rbbox_to_corners(gboxes[:, [0, 2, 3, 5, 6]])
                corners_qboxes = rbbox_to_corners(qboxes[:, [0, 2, 3, 5, 6]])
                # compute the inter area
                # inter_area = rinter_area_compute(corners_gboxes, corners_qboxes)
                # compute the mbr area
                # mbr_area = mbr_area_compute(corners_gboxes, corners_qboxes)
                # compute the inter area
                rinter_area_compute_object = rinter_area_compute()
                inter_area = rinter_area_compute_object(corners_gboxes, corners_qboxes)

                corners_gboxes_1 = torch.stack((corners_gboxes[:, [0, 2, 4, 6]], corners_gboxes[:, [1, 3, 5, 7]]), 2)
                corners_qboxes_1 = torch.stack((corners_qboxes[:, [0, 2, 4, 6]], corners_qboxes[:, [1, 3, 5, 7]]), 2)
                corners_pts = torch.cat((corners_gboxes_1, corners_qboxes_1), 1)

                # compute the mbr area
                mbr_area_compute_object = mbr_area_compute()
                mbr_area = mbr_area_compute_object(corners_pts)

                ## Compute the gious for 3D
                # inter_h = (torch.min(gboxes[:, 2] + gboxes[:, 5], qboxes[:, 2] + qboxes[:, 5]) - torch.max(gboxes[:, 2], qboxes[:, 2]))
                # oniou_h = (torch.max(gboxes[:, 2] + gboxes[:, 5], qboxes[:, 2] + qboxes[:, 5]) - torch.min(gboxes[:, 2], qboxes[:, 2]))
                inter_h = (torch.min(gboxes[:, 1], qboxes[:, 1]) - torch.max(gboxes[:, 1] - gboxes[:, 4],
                                                                             qboxes[:, 1] - qboxes[:, 4]))
                oniou_h = (torch.max(gboxes[:, 1], qboxes[:, 1]) - torch.min(gboxes[:, 1] - gboxes[:, 4],
                                                                             qboxes[:, 1] - qboxes[:, 4]))
                inter_h[inter_h < 0] = 0
                volume_gboxes = gboxes[:, 3].mul(gboxes[:, 4]).mul(gboxes[:, 5])
                volume_qboxes = qboxes[:, 3].mul(qboxes[:, 4]).mul(qboxes[:, 5])
                inter_area_cuda = inter_area.to(torch.device(gboxes.device))
                mbr_area_cuda = mbr_area.to(torch.device(gboxes.device))
                volume_inc = inter_h.mul(inter_area_cuda)
                volume_con = oniou_h.mul(mbr_area_cuda)
                volume_union = (volume_gboxes + volume_qboxes - volume_inc)
                volume_ca = volume_con - volume_union
                ious = torch.div(volume_inc, volume_union)
                ## compute the ious of 2d
                # ious2d =  torch.div(inter_area_cuda, (gboxes[:, 3].mul(gboxes[:, 5]) + qboxes[:, 3].mul(qboxes[:, 5]) - inter_area_cuda))
                gious = torch.zeros([gboxes.shape[0], ], device=gboxes.device, dtype=torch.float32)
                gious[index_loc[:, 0]] = ious[index_loc[:, 0]] - torch.div(volume_ca[index_loc[:, 0]],
                                                                           volume_con[index_loc[:, 0]])
                for i in range(inter_area.shape[0]):
                    # print("ious2d  = ", ious)
                    if (gious[i] < -1):
                        print("infor: (%.4f %.4f %.4f %.4f %.4f %.4f %.4f,%.4f %.4f %.4f %.4f)"
                              % (
                              i, inter_h[i], oniou_h[i], inter_area[i], mbr_area[i], ious[i], gious[i], volume_inc[i],
                              volume_con[i], volume_union[i], volume_ca[i]))

            return torch.unsqueeze(gious, 1)


## For testing purposes
class gious_2d_loss(nn.Module):
    # Compute the inter area of two rotated rectangles
    def _init_(self, gboxes, qboxes):
        super(gious_2d_loss, self)._init_()
        self.gboxes = gboxes
        self.qboxes = qboxes
        return

    def forward(ctx,gboxes, qboxes):

        gious_2D_object = gious_2D()
        gious = gious_2D_object(gboxes, qboxes)
        loss  = torch.sum(gious)
        return loss

class gious_3d_loss(nn.Module):
    # Compute the inter area of two rotated rectangles
    def _init_(self, gboxes, qboxes):
        super(gious_3d_loss, self)._init_()
        self.gboxes = gboxes
        self.qboxes = qboxes
        return

    def forward(ctx,gboxes, qboxes):
        gious_3D_object = gious_3D()
        gious = gious_3D_object(gboxes, qboxes)
        loss  = torch.sum(gious)
        return loss

class corners_loss(nn.Module):
    #
    def _init_(self, gboxes, qboxes):
        super(corners_loss, self)._init_()
        self.gboxes = gboxes
        self.qboxes = qboxes
        return

    def forward(self, gboxes, qboxes):
        # get the corners of the rectangles
        corners_gboxes = rbbox_to_corners(gboxes[:, [0, 2, 3, 5, 6]])
        corners_qboxes = rbbox_to_corners(qboxes[:, [0, 2, 3, 5, 6]])
        loss = torch.sum(torch.abs(corners_gboxes - corners_qboxes))
        return loss

compute_vertex  = compute_vertex.apply
#rbbox_to_corners= rbbox_to_corners.apply
sort_vertex  = sort_vertex.apply
area_polygon = area_polygon.apply
#rinter_area_compute  = rinter_area_compute.apply
#mbr_convex_hull  = mbr_convex_hull.apply
find_convex_hull = find_convex_hull.apply
#mbr_area_compute = mbr_area_compute.apply
#gious_2D = gious_2D.apply
#gious_3D = gious_3D.apply
## For debugging purpose
#corners_loss  = corners_loss.apply
#gious_3d_loss = gious_3d_loss.apply
# Test forward
def test_forward(gboxes, qboxes):
    ious_2d, gious_2d = gious_2D(gboxes[:, [0, 2, 3, 5, 6]], qboxes[:, [0, 2, 3, 5, 6]])
    ious_2d = ious_2d.numpy()
    ious_2d = gious_2d.numpy()

    ious_3d, gious_3d, inter_h = gious_3D(gboxes, qboxes)
    ious_3d = ious_3d.numpy()
    gious_3d = gious_3d.numpy()
    inter_h = inter_h.numpy()

    print("General ious:\n")
    for i in range(N):
        if (ious_3d[i] < 0):
            print("Wrong : Gious: (%.4f %.4f %.4f)" % (ious_3d[i], gious_3d[i], inter_h[i]))
        else:
            print("Gious: (%.4f %.4f %.4f)" % (ious_3d[i], gious_3d[i], inter_h[i]))

if __name__ == '__main__':

    N = 10
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    gboxes       = torch.zeros((N,7),  dtype= torch.float32, device=device)
    qboxes       = torch.zeros((N,7),  dtype= torch.float32, device=device)

    print(gboxes.device)
    print(qboxes.device)


    ious_shapely = np.zeros((N,), dtype= np.float32)
    for i in range(N):
        arr = np.arange(1,100)
        np.random.shuffle(arr)
        cx =arr[3]
        cy =arr[4]
        cz =arr[5]
        wx =arr[6]
        wy =arr[7]
        wz =arr[8]
        a = np.array([cx, cy, cz, wx, wy, wz, np.pi*random.random()*arr[8]])
        arr = arr/5.0
        b = np.array([cx + arr[0], cy+ arr[1], cz+ arr[2], wx+ arr[3], wy+ arr[4], wz+ arr[5], np.pi*random.random()*arr[9]])
        gboxes[i] = torch.from_numpy(a)
        qboxes[i] = torch.from_numpy(b)

        rect1 = np.array([a[0], a[2], a[3], a[5], a[6]])
        rect2 = np.array([b[0], b[2], b[3], b[5], b[6]])
        rect1[4] = rect1[4] * 180.0 / np.pi
        rect2[4] = rect2[4] * 180.0 / np.pi
        #ious_shapely[i] = compute_ious_whih_shapely(rect1, rect2)




    qboxes_varible = torch.autograd.Variable(qboxes, requires_grad=True)
    gboxes_varible = torch.autograd.Variable(gboxes, requires_grad=False)
    rbbox_to_corners_object = rbbox_to_corners()
    corners_gboxes = rbbox_to_corners_object(qboxes_varible[:, [0, 2, 3, 5, 6]])
    corners_qboxes = rbbox_to_corners_object(gboxes_varible[:, [0, 2, 3, 5, 6]])

    corners_gboxes_1 = torch.stack((corners_gboxes[:, [0, 2, 4, 6]], corners_gboxes[:, [1, 3, 5, 7]]), 2)
    corners_qboxes_1 = torch.stack((corners_qboxes[:, [0, 2, 4, 6]], corners_qboxes[:, [1, 3, 5, 7]]), 2)
    corners_pts = torch.cat((corners_gboxes_1, corners_qboxes_1), 1)

    # compute the mbr area
    #mbr_area_compute_object = mbr_area_compute()
    #mbr_area = mbr_area_compute_object(corners_pts)

    # compute the inter area
    # inter_area = rinter_area_compute(corners_gboxes, corners_qboxes)
    # compute the mbr area
    # mbr_area = mbr_area_compute(corners_gboxes, corners_qboxes)
    # compute the inter area
    #rinter_area_compute_object = rinter_area_compute()
    #inter_area = rinter_area_compute_object(corners_gboxes, corners_qboxes)

    #N = corners_pts.shape[0]
    # mbr_rect_areas   = torch.zeros((N,), dtype=torch.float32)
   # mbr_rect_area = []

    # mbr_rect_areas   = torch.zeros((N,), dtype=torch.float32, device = corners_gboxes.device)
    #for iter in range(N):
    #    convex_hull_pts = find_convex_hull(corners_pts[iter, :, :].squeeze())

    #    res = torch.autograd.gradcheck(mbr_convex_hull(), (convex_hull_pts), eps=1e-2, atol=1e-2, raise_exception=True)
    #    print(res)



    aligned = True
    res = torch.autograd.gradcheck(gious_3D(), (gboxes_varible, qboxes_varible, aligned), eps=1e-2, atol=1e-2, raise_exception=True)
    print(res)  # res should be True if the gradients are correct.







    '''
    object_gious_2d_loss = gious_2d_loss()
    gious_2d_losss = object_gious_2d_loss(gboxes, qboxes_varible)
    print("gious_2d_loss=", gious_2d_losss)
    gious_2d_losss.backward()
    print("qboxes_varible.grad=", qboxes_varible.grad)
    
    '''
    '''
    object_gious_3d_loss = gious_3d_loss()
    gious_3d_losss = object_gious_3d_loss(gboxes, qboxes_varible)
    print("gious_3d_loss=", gious_3d_losss)
    gious_3d_losss.backward()
    print("qboxes_varible.grad=", qboxes_varible.grad)
    '''
    '''    
    object_corners_loss = corners_loss()
    corners_losses = object_corners_loss(gboxes, qboxes_varible)
    corners_losses.backward()
    print("First loss=", corners_losses)
    print("qboxes_varible.grad=", qboxes_varible.grad)
    '''
    #gious_loss = gious_3d_loss(gboxes, qboxes_varible)
    #gious_loss.backward()






















