import lsst.daf.butler as dafButler
import sys
sys.path.append("/sdf/data/rubin/shared/image_quality/imsim/notebooks/")
from collection_dictionary_shared import collection_dictionary

# Import packages for  Catalog Access
import pandas
pandas.set_option('display.max_rows', 1000)
from lsst.rsp import get_tap_service, retrieve_query

def fits_from_calexp(seqnum, det, repofolder, folderout, visit_base = 5023071800000):
    
    collection_dict = collection_dictionary()
    visit_seqnum = visit_base + seqnum
    collection = collection_dict[seqnum]

    butler = dafButler.Butler(repofolder, collections=collection)

    datasetType='calexp'
    dataId = {'visit': visit_seqnum, 'detector': det, 'band':'r'}
    calexp = butler.get(datasetType, **dataId)

    calexp_image = calexp.image
    calexp_image.writeFits(folderout+'imsim_fits_from_calexp_seqnum{:04d}_det{:03d}'.format(seqnum, det)+'.fits')
    