# Diabetic_Retinopathy_Detection
Rapid Screening and Early diagnosis of Disease Retinopathy using Scanned Retinal images and underlying Deep Learning models.

<img src='https://static01.nyt.com/images/2019/03/10/business/10aieyedoctor1/10aidoctor1-facebookJumbo.jpg'/>

<h3>What is Diabetic Retinopathy?</h3>

Diabetic retinopathy is the most common form of diabetic eye disease and usually affects people who have diabetes for a significant number of years. The risk of diabetic eye is for aged people especially working persons in rural and slum areas.
It increase with age as well as with less well controlled blood sugar and blood pressure level and occurs when the damaged blood vessels leak blood and other fluids into your retina, causing swelling and blurry vision. The blood vessels can become blocked, scar tissue can develop, and retinal detachment can eventually occur.

<img src='https://cdn-images-1.medium.com/max/1800/0*d_uPvD9xdpW7qxYE.png'>

<h3>Problem Statement:</h3>

Technicians of Aravind Eye Hospital has collected large number of scanned retinal images of diabetic persons by travelling through rural areas and hosted the problem in kaggle as a competition where the best solutions will be spread to other ophthalmologists through APTOS.
They want us to build a system where it takes the retinal image of a patient and tells us the severity of diabetic retinopathy.

<h3>The final solution to our problem statement</h3>

* After applying couple of pretrained models, Ensembling those models has given a top score of 0.9314.
* But here our goal is not only to give best score but to provide a system which makes prediction with explanation. Here I have chosen DenseNet which has given next best score of 0.923 val kappa score. 
* Using Grad-Cam for model interpretability.


