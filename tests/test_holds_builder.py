"""D's hold map builder (coach/server/holds.py) driving A's real EditorState, Roboflow faked."""

import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from coach.bus import Bus
from coach.perception import holdmap_editor
from coach.perception.holds import load_holdmap_with_shapes
from coach.server import holds as holds_module
from coach.server.app import create_app
from coach.server.holds import HoldMapBuilder, camera_shift, features, same_view

W, H = 300, 200
# Three yellow route holds low to high, and one blue hold that isn't on the route.
ROUTE = [(60, 150), (80, 100), (100, 50)]
BLUE = (200, 100)
BOXES = [(50, 140, 71, 161), (70, 90, 91, 111), (90, 40, 111, 61), (190, 90, 211, 111)]
CFG = {
    "detector_model": "fake/1",
    "api_url": "https://example.invalid",
    "hue_tolerance": 12,
    "min_saturation": 60,
    "white_min_value": 210,
    "white_max_saturation": 55,
    "white_min_fraction": 0.15,
    "patch_radius_px": 6,
    "morph_kernel_px": 5,
    "min_area": 0.0003,
    "max_area": 0.02,
    "split_peak_ratio": 0.75,
    "start_count": 2,
    "drag_min_px": 6,
    "polygon_epsilon": 0.02,
}


def bgr(hue):
    hsv = np.uint8([[[hue, 200, 220]]])
    return tuple(int(v) for v in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


def wall(width=W, height=H):
    img = np.full((height, width, 3), (120, 120, 120), np.uint8)
    for cx, cy in ROUTE:
        cv2.circle(img, (cx, cy), 10, bgr(30), -1)
    cv2.circle(img, BLUE, 10, bgr(110), -1)
    return img


def norm(x, y):
    return {"x": x / (W - 1), "y": y / (H - 1)}


class FakeControl:
    def __init__(self, image=None, pictures=None, frame=None, video_s=None) -> None:
        self.image = wall() if image is None else image
        self.pictures = pictures or {}  # more stills: key -> picture
        self.frame = frame  # the video's newest frame
        self.video_s = video_s  # also offer a "clip" still this many seconds long
        self.problem_id = "p1"
        self.reloads = self.resumes = 0
        self.paused = False
        self.reads: list[tuple] = []  # (key, at) of every read_still

    def listen(self, on_hello) -> None:
        pass

    def stills(self) -> list[dict]:
        out = [
            {"key": k, "name": k, **({"live": True} if k == "frame" else {})} for k in self.pictures
        ] or [{"key": "wall", "name": "Empty wall"}]
        if self.video_s:
            out.append({"key": "clip", "name": "Clip", "duration": self.video_s, "at": 1.0})
        return out

    def read_still(self, key: str, at: float | None = None):
        self.reads.append((key, at))
        if key == "clip" and self.video_s:
            img = self.image.copy()
            img[0, 0] = round(at * 10)  # which moment, readable from the picture
            return img
        if self.pictures and key in self.pictures:
            return self.pictures[key]
        if not self.pictures and key == "wall":
            return self.image
        raise ValueError(f"not an offered still: {key!r}")

    def latest_frame(self):
        return self.frame

    async def pause(self) -> None:
        self.paused = True

    async def resume(self) -> dict:
        self.paused = False
        self.resumes += 1
        return {"session": 2, "problem_id": self.problem_id}

    async def reload(self) -> dict:
        self.paused = False
        self.reloads += 1
        return {"session": 2, "problem_id": self.problem_id}


def scene():
    """A busy wall wider than one picture, so a camera move can be cut out of it."""
    rng = np.random.default_rng(7)
    img = np.full((600, 440, 3), 190, np.uint8)
    for _ in range(400):
        centre = (int(rng.integers(0, 440)), int(rng.integers(0, 600)))
        colour = tuple(int(c) for c in rng.integers(0, 256, 3))
        cv2.circle(img, centre, int(rng.integers(3, 18)), colour, -1)
    return img


def views():
    """The video's frame, a still from the same spot (other exposure), one 10% further right."""
    s = scene()
    frame = np.ascontiguousarray(s[:, :400])
    same = cv2.convertScaleAbs(frame, alpha=0.9, beta=12)
    moved = np.ascontiguousarray(s[:, 40:])
    return frame, same, moved


@pytest.fixture
def fake_roboflow(monkeypatch):
    polygons = {BOXES[1]: [(70, 100), (80, 90), (91, 100), (80, 111)]}  # one diagonal hold
    monkeypatch.setattr(holdmap_editor, "detect_holds", lambda _img, _cfg: list(BOXES))
    monkeypatch.setattr(holdmap_editor, "detect_hold_polygons", lambda _img, _cfg: polygons)


@pytest.fixture
def builder(tmp_path, fake_roboflow):
    b = HoldMapBuilder(FakeControl(), root=tmp_path, uploads=tmp_path / "stills")
    b.parts().load_cfg = lambda: CFG
    return b


async def test_open_shows_the_still_and_asks_for_the_route_colour(builder):
    view = await builder.open("wall")
    assert view["open"] and view["size"] == [W, H] and view["holds"] == []
    assert view["message"].startswith("Click a hold of the route's colour")
    assert builder.jpeg[:2] == b"\xff\xd8"  # a JPEG
    assert not builder.status()["has_map"]


async def test_picking_a_colour_keeps_that_colours_holds_numbered_bottom_to_top(builder):
    await builder.open("wall")
    view = await builder.op("pick", norm(*ROUTE[0]))
    kept = [h for h in view["holds"] if h["kept"]]
    assert view["detector"] and view["colour"] == "yellow" and view["kept"] == 3
    assert view["message"].startswith("Found 3 yellow holds")
    assert [(h["id"], h["role"]) for h in sorted(kept, key=lambda h: h["id"])] == [
        (1, "start"),
        (2, "start"),
        (3, "top"),
    ]
    lowest = next(h for h in kept if h["id"] == 1)
    assert lowest["box"] == pytest.approx([50 / W, 140 / H, 71 / W, 161 / H])
    assert next(h for h in kept if h["id"] == 2)["polygon"] is not None  # outline kept


async def test_a_missed_colour_click_says_so_and_keeps_the_holds_found(builder):
    await builder.open("wall")
    await builder.op("pick", norm(*ROUTE[0]))
    view = await builder.op("pick", norm(250, 20))  # bare grey wall
    assert view["message"].startswith("Couldn't find holds: no coloured or white pixels")
    assert view["kept"] == 3


async def test_roboflow_failure_is_reported_in_plain_words(builder, monkeypatch):
    def offline(*_args):
        raise RuntimeError("ROBOFLOW_API_KEY is not set")

    monkeypatch.setattr(holdmap_editor, "detect_holds", offline)
    await builder.open("wall")
    view = await builder.op("pick", norm(*ROUTE[0]))
    assert view["message"] == (
        "Couldn't find holds: Roboflow unavailable: ROBOFLOW_API_KEY is not set. "
        "Drag with Add box to place holds by hand."
    )
    assert "use r" not in view["message"]  # the desktop editor's key hints are dropped


async def test_leave_out_put_back_mark_erase_add_and_undo(builder):
    await builder.open("wall")
    await builder.op("pick", norm(*ROUTE[0]))
    assert (await builder.op("toggle", norm(*ROUTE[2])))["message"] == "Left a hold out."
    view = await builder.op("toggle", norm(*ROUTE[2]))
    assert view["message"] == "Put a hold back." and view["kept"] == 3
    assert (await builder.op("toggle", norm(250, 20)))["message"].startswith("There's no hold")
    assert (await builder.op("top", norm(*ROUTE[1])))["message"] == "Marked a top hold."
    assert (await builder.op("start", norm(*BLUE)))["message"].startswith("Click a kept hold")

    view = await builder.op("erase", {"x0": 0.15, "y0": 0.65, "x1": 0.25, "y1": 0.85})
    assert view["message"] == "Erased 1 hold." and view["kept"] == 2
    view = await builder.op("box", {"x0": 0.8, "y0": 0.1, "x1": 0.9, "y1": 0.2})
    assert view["message"] == "Added a hold." and view["kept"] == 3
    assert (await builder.op("box", {"x0": 0.5, "y0": 0.5, "x1": 0.5, "y1": 0.5}))[
        "message"
    ] == "Drag a bigger box to add a hold."
    assert (await builder.op("undo", {}))["message"] == "Undone."
    assert (await builder.op("undo", {}))["kept"] == 3  # the erase is undone too


async def test_bad_ops_and_coordinates_are_rejected(builder):
    with pytest.raises(ValueError):
        await builder.op("pick", norm(*ROUTE[0]))  # nothing open yet
    await builder.open("wall")
    with pytest.raises(ValueError):
        await builder.op("rotate", {})
    with pytest.raises(ValueError):
        await builder.op("toggle", {"x": 1.5, "y": 0.5})
    with pytest.raises(ValueError):
        await builder.open("nope")


async def test_save_writes_the_map_and_restarts_analysis(builder, tmp_path):
    await builder.open("wall")
    with pytest.raises(ValueError, match="no holds"):
        await builder.save()
    await builder.op("pick", norm(*ROUTE[0]))
    result = await builder.save()

    holdmap, shapes = load_holdmap_with_shapes("p1", tmp_path)
    assert result["holds"] == 3 and result["hello"]["session"] == 2
    assert builder.control.reloads == 1 and builder.editor is None
    assert [h.role for h in holdmap.holds] == ["start", "start", "top"]
    assert list(shapes) == [2] and holdmap.image_size == (W, H)
    assert builder.shapes()["shapes"]["2"][0] == pytest.approx([70 / W, 100 / H])


async def test_reopening_edits_the_saved_map_only_for_a_picture_of_the_same_shape(
    builder, tmp_path
):
    await builder.open("wall")
    await builder.op("pick", norm(*ROUTE[0]))
    await builder.save()

    view = await builder.open("wall")
    assert view["kept"] == 3 and view["message"].startswith("Editing the saved map (3 holds)")
    assert view["origin"] == "from the saved map"

    builder.control.image = wall(H, W)  # a portrait picture: the saved boxes wouldn't line up
    view = await builder.open("wall")
    assert view["kept"] == 0 and "doesn't match this one" in view["message"]


def test_http_endpoints(tmp_path, fake_roboflow):
    control = FakeControl()
    app = create_app(Bus(), control=control)
    with TestClient(app) as client:
        app.state.holds.root = tmp_path
        app.state.holds.uploads = tmp_path / "stills"
        app.state.holds.parts().load_cfg = lambda: CFG

        status = client.get("/api/holds").json()
        assert status["available"] and status["problem_id"] == "p1" and status["last"] is None
        assert status["stills"] == [{"key": "wall", "name": "Empty wall", "fits": None}]
        assert client.get("/api/holds/still.jpg").status_code == 404  # nothing open yet
        assert client.get("/api/holds/view").json() == {"open": False}

        assert client.post("/api/holds/open", json={"still": "wall"}).json()["open"]
        assert client.get("/api/holds/view").json()["still"] == "wall"  # a reloaded page's way back
        assert control.paused
        still = client.get("/api/holds/still.jpg")
        assert still.headers["content-type"] == "image/jpeg"
        view = client.post("/api/holds/op", json={"op": "pick", **norm(*ROUTE[0])}).json()
        assert view["kept"] == 3
        assert client.post("/api/holds/op", json={"op": "rotate"}).status_code == 409
        saved = client.post("/api/holds/save").json()
        assert saved["holds"] == 3 and control.reloads == 1
        assert json.loads((tmp_path / "p1.json").read_text())["problem_id"] == "p1"
        assert client.get("/api/holds/shapes").json()["shapes"].keys() == {"2"}
        assert client.post("/api/holds/close").json() == {"open": False}
        assert control.resumes == 0  # saving restarted analysis already

        client.post("/api/holds/open", json={"still": "wall"})
        assert client.post("/api/holds/close").json() == {"open": False}
        assert control.resumes == 1 and not control.paused
        assert client.get("/api/holds").json()["last"] == {"key": "wall", "at": None}

        # Your own picture: the raw JPEG in the body, its file name in the query.
        jpeg = cv2.imencode(".jpg", wall())[1].tobytes()
        view = client.post(
            "/api/holds/upload?name=wall photo.jpg",
            content=jpeg,
            headers={"content-type": "image/jpeg"},
        ).json()
        assert view["still"] == "upload:wall-photo.jpg" and view["size"] == [W, H]
        assert view["duration"] is None and control.paused
        bad = client.post("/api/holds/upload?name=x.jpg", content=b"not a picture")
        assert bad.status_code == 409 and "isn't a picture" in bad.json()["detail"]
        client.post("/api/holds/close")

        control.video_s = 4.0  # a video still: opened at a chosen moment
        view = client.post("/api/holds/open", json={"still": "clip", "at": 2.5}).json()
        assert view["at"] == 2.5 and view["duration"] == 4.0
        assert client.post("/api/holds/op", json={"op": "frame", "at": 3}).json()["at"] == 3.0


def test_hold_endpoints_404_without_a_holds_control():
    with TestClient(create_app(Bus())) as client:
        assert client.get("/api/holds").status_code == 404
        assert client.post("/api/holds/save").status_code == 404


async def test_marking_one_role_keeps_the_automatic_others(builder):
    await builder.open("wall")
    await builder.op("pick", norm(*ROUTE[0]))  # automatic: 1, 2 start; 3 top
    view = await builder.op("top", norm(*ROUTE[1]))  # hold 2 becomes the top instead
    roles = {h["id"]: h["role"] for h in view["holds"] if h["kept"]}
    assert roles == {1: "start", 2: "top", 3: "mid"}  # the automatic start 1 stayed

    await builder.open("wall")
    await builder.op("pick", norm(*ROUTE[0]))
    view = await builder.op("start", norm(*ROUTE[1]))  # unmark automatic start 2
    roles = {h["id"]: h["role"] for h in view["holds"] if h["kept"]}
    assert roles == {1: "start", 2: "mid", 3: "top"}  # the automatic top stayed


# ---- Picking the colour again ----------------------------------------------------------------


async def test_picking_the_colour_again_replaces_the_holds_instead_of_adding_to_them(builder):
    await builder.open("wall")
    await builder.op("pick", norm(*ROUTE[0]))
    await builder.save()
    view = await builder.open("wall")  # the saved map: its 3 holds become editable ones
    assert view["kept"] == 3 and view["origin"] == "from the saved map"

    view = await builder.op("recolour", {})
    assert view["picking"] and view["kept"] == 3  # nothing changes until a hold is clicked
    view = await builder.op("pick", norm(*ROUTE[0]))

    assert view["kept"] == 3  # was 6: the saved holds plus the same holds found again
    assert view["message"].startswith("Started over with 3 yellow holds")
    assert not view["picking"] and view["origin"] == "found by Roboflow"
    assert (await builder.op("pick", norm(*BLUE)))["kept"] == 1  # another colour: only its hold

    view = await builder.op("undo", {})
    assert view["kept"] == 3 and view["colour"] == "yellow"
    view = await builder.op("undo", {})  # back past both picks to the saved map
    assert view["kept"] == 3 and view["origin"] == "from the saved map"
    assert view["message"] == "Undone: the holds from before you picked the colour are back."
    assert (await builder.op("undo", {}))["message"] == "Nothing to undo."


async def test_a_missed_click_after_pick_the_colour_again_keeps_waiting(builder):
    await builder.open("wall")
    await builder.op("pick", norm(*ROUTE[0]))
    await builder.op("toggle", norm(*ROUTE[2]))
    await builder.op("recolour", {})

    view = await builder.op("pick", norm(250, 20))  # bare wall
    assert view["picking"] and view["kept"] == 2 and view["message"].startswith("Couldn't find")
    view = await builder.op("keep", {})
    assert not view["picking"] and view["kept"] == 2  # the edits survived
    assert (await builder.op("undo", {}))["kept"] == 3  # and their undo too


# ---- Analysis pauses while the editor is open -------------------------------------------------


async def test_analysis_pauses_while_the_editor_is_open(builder):
    control = builder.control
    with pytest.raises(ValueError):
        await builder.open("nope")
    assert not control.paused  # a picture that can't be read doesn't pause anything

    await builder.open("wall")
    assert control.paused
    await builder.close()
    assert not control.paused and control.resumes == 1
    await builder.close()  # nothing open: nothing to restart
    assert control.resumes == 1

    await builder.open("wall")
    await builder.op("pick", norm(*ROUTE[0]))
    await builder.save()  # saving restarts analysis with the new map instead
    assert not control.paused and control.reloads == 1 and control.resumes == 1


# ---- Stills filmed from the video's camera position -------------------------------------------


def test_camera_shift_tells_a_moved_camera_from_the_same_spot():
    frame, same, moved = views()
    video = features(frame)
    assert camera_shift(features(same), video) < 0.01
    assert camera_shift(features(moved), video) == pytest.approx(0.1, abs=0.01)
    assert same_view(features(same), video) is True
    assert same_view(features(moved), video) is False
    assert same_view(features(np.ascontiguousarray(frame.transpose(1, 0, 2))), video) is False
    assert same_view(features(np.full_like(frame, 128)), video) is None  # nothing to match


async def test_stills_from_the_videos_camera_spot_come_first(tmp_path, fake_roboflow):
    frame, same, moved = views()
    control = FakeControl(pictures={"moved": moved, "same": same, "frame": frame})
    builder = HoldMapBuilder(control, root=tmp_path, uploads=tmp_path / "stills")
    builder.parts().load_cfg = lambda: CFG

    fits = [(s["key"], s["fits"]) for s in builder.stills()]
    assert fits == [("frame", True), ("moved", None), ("same", None)]  # no video yet: can't tell

    control.frame = frame
    fits = [(s["key"], s["fits"]) for s in builder.stills()]
    assert fits == [("same", True), ("frame", True), ("moved", False)]
    assert builder.status()["stills"][0]["key"] == "same"  # the editor opens this one

    assert (await builder.open("moved"))["fits"] is False
    assert (await builder.open("same"))["fits"] is True


# ---- Any moment of a video ---------------------------------------------------------------------


async def test_a_video_still_opens_at_its_default_moment_or_a_chosen_one(builder):
    builder.control.video_s = 4.0
    view = await builder.open("clip")
    assert view["at"] == 1.0 and view["duration"] == 4.0  # the control's default moment
    assert builder.control.reads[-1] == ("clip", 1.0)

    view = await builder.open("clip", at=9.0)
    assert view["at"] == 4.0  # kept inside the clip
    assert (await builder.open("wall"))["at"] is None  # a picture has no moments


async def test_moving_to_another_moment_keeps_the_holds_and_shows_the_new_frame(builder):
    builder.control.video_s = 4.0
    first = await builder.open("clip")
    await builder.op("pick", norm(*ROUTE[0]))
    jpeg = builder.jpeg

    view = await builder.op("frame", {"at": 2.5})

    assert view["kept"] == 3 and view["at"] == 2.5 and view["version"] == first["version"] + 1
    assert view["message"] == "Showing 2.5 s into the video. The holds stay where they are."
    assert builder.editor.image[0, 0, 0] == 25 and builder.jpeg != jpeg  # the new frame
    assert builder.status()["last"] == {"key": "clip", "at": 2.5}  # reopening comes back here

    await builder.op("frame", {"at": 3.0})
    await builder.op("undo", {})  # back past the pick to the empty editor...
    assert builder.editor.image[0, 0, 0] == 30  # ...still showing the frame on screen


async def test_only_a_video_still_has_moments(builder):
    await builder.open("wall")
    with pytest.raises(ValueError, match="isn't from a video"):
        await builder.op("frame", {"at": 1.0})


# ---- Your own pictures ---------------------------------------------------------------------------


def png(img) -> bytes:
    return cv2.imencode(".png", img)[1].tobytes()


async def test_an_uploaded_picture_is_kept_offered_and_opened(builder, tmp_path):
    view = await builder.upload("../My wall (2).PNG", png(wall()))

    kept = tmp_path / "stills" / "p1" / "My-wall-2.jpg"  # no paths, plain name, always a JPEG
    assert kept.exists() and view["still"] == "upload:My-wall-2.jpg"
    assert view["open"] and view["size"] == [W, H] and view["duration"] is None
    assert builder.control.paused  # opened like any still
    assert (await builder.op("pick", norm(*ROUTE[0])))["kept"] == 3  # Roboflow sees it

    names = [s["name"] for s in builder.stills()]
    assert names == ["Empty wall", "Your picture · My-wall-2.jpg"]
    assert (await builder.open("upload:My-wall-2.jpg"))["size"] == [W, H]  # read back from disk


async def test_uploads_that_arent_usable_pictures_are_refused(builder, monkeypatch):
    with pytest.raises(ValueError, match="isn't a picture"):
        await builder.upload("notes.txt", b"just some text")
    with pytest.raises(ValueError, match="too small"):
        await builder.upload("icon.png", png(np.zeros((20, 20, 3), np.uint8)))
    monkeypatch.setattr(holds_module, "UPLOAD_MAX_BYTES", 100)
    with pytest.raises(ValueError, match="over"):
        await builder.upload("huge.png", png(wall()))
    with pytest.raises(ValueError, match="not an offered still"):
        await builder.open("upload:../../p1.json")  # only listed pictures can be opened
    assert not builder.control.paused and builder.editor is None
