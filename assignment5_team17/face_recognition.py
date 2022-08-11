from EigenFaces import *

main_matrix = Existed_Images()
Covariance_Matrix, mean_image, a = calculate_covariance(main_matrix)
eigenfaces, counter = calculate_eigenfaces(Covariance_Matrix)

path = "Test_Dataset\Kayla-Person.jpg"
projected_img = proj_test_img(path, eigenfaces, mean_image, counter)
class_of_img, dist , list_of_dist = calculate_similarity(eigenfaces, projected_img, a, counter)

print(f"Class :{class_of_img}")
print(f"Distance :{dist}")
