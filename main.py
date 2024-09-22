import demo

input_path = './testni_posnetki/testna_4'
task = demo.ModelTask.DETECT_OBSTACLES  # / DETECT_DOORS / CALIB_ALIGN / CALIB_CAMERA
output_path = None # '<optional output folder path>'

# Vhod, način delovanja, (opcijski) izhod
demo.main(input_path=input_path, task=task, output_path=output_path)
