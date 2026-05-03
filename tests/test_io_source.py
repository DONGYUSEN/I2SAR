import tarfile
import zipfile

from i2sar.io.base import SourceRef
from i2sar.io.source import build_member_ref, list_product_members


def test_source_ref_builds_directory_and_vsi_paths(tmp_path):
    directory_file = tmp_path / "product" / "scene.tiff"
    directory_file.parent.mkdir()
    directory_file.write_bytes(b"")

    directory_ref = SourceRef(path=str(directory_file), storage="file")
    zip_ref = SourceRef(path=str(tmp_path / "product.zip"), storage="zip", member="measurement/scene.tiff")
    tar_ref = SourceRef(path=str(tmp_path / "product.tar"), storage="tar", member="measurement/scene.tiff")

    assert directory_ref.vsi_path() == str(directory_file)
    assert zip_ref.vsi_path().startswith("/vsizip/")
    assert zip_ref.vsi_path().endswith("/measurement/scene.tiff")
    assert tar_ref.vsi_path().startswith("/vsitar/")


def test_list_product_members_supports_directory_zip_and_tar(tmp_path):
    product = tmp_path / "product"
    (product / "annotation").mkdir(parents=True)
    (product / "annotation" / "scene.xml").write_text("<root/>", encoding="utf-8")

    zip_path = tmp_path / "product.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("annotation/scene.xml", "<root/>")

    tar_path = tmp_path / "product.tar"
    source_xml = tmp_path / "scene.xml"
    source_xml.write_text("<root/>", encoding="utf-8")
    with tarfile.open(tar_path, "w") as tf:
        tf.add(source_xml, arcname="annotation/scene.xml")

    assert list_product_members(product) == ["annotation/scene.xml"]
    assert list_product_members(zip_path) == ["annotation/scene.xml"]
    assert list_product_members(tar_path) == ["annotation/scene.xml"]


def test_build_member_ref_records_storage_and_member(tmp_path):
    zip_path = tmp_path / "product.zip"
    ref = build_member_ref(zip_path, "measurement/scene.tiff")

    assert ref.storage == "zip"
    assert ref.member == "measurement/scene.tiff"
