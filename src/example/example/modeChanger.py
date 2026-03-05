from .systemMode import CarMode, CarSpeed
import time

class StateChanger:
    def __init__(self):
        self.cur_mode = CarMode.STRAIGHT
        self.cur_speed = CarSpeed.NORMAL

        self.stop_halt = False
        self.timer = 0.0  # Elapsed time since creation, ends at termination
        self.time_checkpoint = 0.0 # Moment of important events, shared with all occurences in modeChanger

        self.lookup_yaw_diffs = []
        self.classes = [
            'pedestrian',
            'cyclist',
            'car',
            'bus',
            'truck',
            'red_light',
            'yellow_light',
            'green_light',
            'crosswalk_sign',
            'enter_highway_sign',
            'leave_highway_sign',
            'oneway_sign',
            'parking_sign',
            'priority_sign',
            'noentry_sign',
            'roundabout_sign',
            'stop_sign'
        ]
        self.idx_to_cls = {
            0: 'pedestrian',
            1: 'cyclist',
            2: 'car',
            3: 'bus',
            4: 'truck',
            5: 'red_light',
            6: 'yellow_light',
            7: 'green_light',
            8: 'crosswalk_sign',
            9: 'enter_highway_sign',
            10: 'leave_highway_sign',
            11: 'oneway_sign',
            12: 'parking_sign',
            13: 'priority_sign',
            14: 'noentry_sign',
            15: 'roundabout_sign',
            16: 'stop_sign'
        }
        # Detection-count thresholds: signs need more consistent detections
        # to avoid triggering from a single noisy frame.
        self.det_threshold = {
            'pedestrian': 3,
            'cyclist': 3,
            'car': 3,
            'bus': 3,
            'truck': 3,
            'red_light': 3,
            'yellow_light': 3,
            'green_light': 3,
            'crosswalk_sign': 5,
            'enter_highway_sign': 5,
            'leave_highway_sign': 5,
            'oneway_sign': 5,
            'parking_sign': 5,
            'priority_sign': 5,
            'noentry_sign': 5,
            'roundabout_sign': 5,
            'stop_sign': 5
        }
        self.cur_dets = {key: 0 for key in self.classes}

        # ── Latched state ─────────────────────────────────────────────
        # Sign-triggered speed/mode persist until another sign overrides
        # them.  Dynamic detections (pedestrian, red light, …) still
        # override temporarily but the latch is the fallback.
        self._latched_speed: CarSpeed = CarSpeed.NORMAL
        self._latched_mode: CarMode = CarMode.STRAIGHT

    def record_detection(self, idxes, boxes):
        def get_max_cnt(cls, coeff=1):
            return int(self.det_threshold[cls] * coeff)
        
        def significant_sign(box, aspect_ratio, err_rate, area_threshold):
            """Return True when the bounding box matches the expected aspect
            ratio (within *err_rate* tolerance) AND the normalised area is
            at least *area_threshold*."""
            aspect_ratio_met =  aspect_ratio * (1 - err_rate) <= box[-2] / box[-1] <= aspect_ratio * (1 + err_rate)
            area_met = box[-2] * box[-1] >= area_threshold
            return aspect_ratio_met and area_met

        # Signs that set a persistent (latched) driving mode — they need
        # a stricter area gate so far-away detections are ignored.
        _SIGN_AREA_THRESHOLD = 0.012   # ~3.5 % of image  ≈ sign is close
        _SIGN_HIGHWAY_AREA   = 0.008   # highway signs are taller / thinner

        dets = [self.idx_to_cls[i] for i in idxes]
        accepted_dets = []
        for d, b in zip(dets, boxes):
            if d in ['pedestrian', 'cyclist'] and significant_sign(b, 1/1, 0.5, 0.006):
                accepted_dets.append(d)
            elif d in ['car', 'bus', 'truck'] and significant_sign(b, 1/1, 0.7, 0.004):
                accepted_dets.append(d)
            elif d in self.classes[5:7] and significant_sign(b, 1/3.5, 0.9, 0.002):
                # Traffic lights — keep original area gate (small but visible)
                accepted_dets.append(d)
            elif d in self.classes[7:]:
                if d in ['enter_highway_sign', 'leave_highway_sign'] and significant_sign(b, 2/3, 0.9, _SIGN_HIGHWAY_AREA):
                    accepted_dets.append(d)
                elif significant_sign(b, 1/1, 0.9, _SIGN_AREA_THRESHOLD):
                    accepted_dets.append(d)

        for c in list(self.cur_dets.keys()):
            if c in accepted_dets:
                self.cur_dets[c] = min(self.cur_dets[c] + 1, get_max_cnt(c))
            else:
                # Signs decay slowly (-1) so the counter doesn't collapse
                # the moment the sign leaves the frame.  Dynamic objects
                # (pedestrians, vehicles, lights) decay faster (-2).
                if c in self.classes[8:]:     # index 8+ are all *_sign
                    self.cur_dets[c] = max(self.cur_dets[c] - 1, 0)
                else:
                    self.cur_dets[c] = max(self.cur_dets[c] - 2, 0)

    def record_lookup(self, yaw_diffs):
        self.lookup_yaw_diffs = yaw_diffs
    
    def update_timer(self, dt):
        self.timer += dt

    def change_state(self):
        '''Handles changes based on the detection recorder.

        Sign-triggered states (highway, roundabout, oneway …) are
        **latched**: they persist until a *different* sign overrides them.
        Dynamic detections (pedestrian, red light, vehicles …) can still
        override temporarily; once they clear the latched state takes
        effect again.
        '''
        # threshold check util
        def threshold_met(cls):
            return self.cur_dets[cls] >= self.det_threshold[cls]

        def threshold_met_contextual(cls, override_threshold):
            """Like threshold_met but uses a lower count when context
            justifies it (e.g. expecting a sign at high speed)."""
            return self.cur_dets[cls] >= override_threshold

        def turn_met(self):
            if len(self.lookup_yaw_diffs) == 0:
                return False
            return max(self.lookup_yaw_diffs) > 30

        # ── 1. Update latched speed/mode when a sign threshold is met ──
        #    These writes only happen the moment the counter crosses the
        #    threshold. The latch survives long after the sign is no
        #    longer visible.
        #
        #    Contextual thresholds: when already on the highway (FAST),
        #    the car passes the leave_highway sign very quickly. Use a
        #    reduced threshold (2) so that even a brief sighting triggers.
        _LEAVE_HWY_THRESH = 2 if self._latched_speed == CarSpeed.FAST else self.det_threshold['leave_highway_sign']

        if threshold_met('enter_highway_sign'):
            self._latched_speed = CarSpeed.FAST
        if threshold_met_contextual('leave_highway_sign', _LEAVE_HWY_THRESH):
            self._latched_speed = CarSpeed.NORMAL
        if threshold_met('oneway_sign'):
            self._latched_speed = CarSpeed.NORMAL
            self._latched_mode = CarMode.STRAIGHT
        if threshold_met('roundabout_sign'):
            self._latched_mode = CarMode.TURN
        if threshold_met('parking_sign'):
            self._latched_mode = CarMode.PARKING

        # ── 2. Speed handling (priority order: highest-priority first) ─
        if self.stop_halt:
            if self.timer < self.time_checkpoint + 1.0:  # set back to 3.0 irl
                self.cur_speed = CarSpeed.STOP
            elif self.time_checkpoint + 1.0 <= self.timer < self.time_checkpoint + 1.5:
                self.cur_speed = CarSpeed.NORMAL
            else:
                self.stop_halt = False
                self.cur_dets['stop_sign'] = 0
                self.cur_speed = CarSpeed.NORMAL
            # NOTE: don't return here — still evaluate mode below
        elif threshold_met('pedestrian') or threshold_met('cyclist'):
            self.cur_speed = CarSpeed.STOP
        elif threshold_met('red_light'):
            self.cur_speed = CarSpeed.STOP
        elif threshold_met('stop_sign') and not self.stop_halt:
            self.cur_speed = CarSpeed.STOP
            self.stop_halt = True
            self.time_checkpoint = self.timer
        elif threshold_met('noentry_sign'):
            self.cur_speed = CarSpeed.STOP
        elif threshold_met('car') or threshold_met('truck') or threshold_met('bus'):
            self.cur_speed = CarSpeed.STOP if self._get_mode() == CarMode.TURN else CarSpeed.NORMAL
        elif threshold_met('yellow_light') and self._get_speed() != CarSpeed.STOP:
            self.cur_speed = CarSpeed.SLOW
        elif threshold_met('crosswalk_sign'):
            self.cur_speed = CarSpeed.SLOW
        elif threshold_met('green_light'):
            self.cur_speed = CarSpeed.NORMAL
        else:
            # No dynamic detection overriding — fall back to latched speed
            self.cur_speed = self._latched_speed

        # ── 3. Mode handling ───────────────────────────────────────────
        if turn_met(self):
            self.cur_mode = CarMode.TURN
        elif threshold_met('car') or threshold_met('truck') or threshold_met('bus'):
            self.cur_mode = CarMode.TAILING if self._get_speed() == CarSpeed.SLOW else CarMode.OVERTAKING
        else:
            # No dynamic detection overriding — fall back to latched mode
            self.cur_mode = self._latched_mode

    def _get_mode(self):
        return self.cur_mode

    def _get_speed(self):
        return self.cur_speed

# EXAMPLE AS BELOW
# if __name__ == '__main__':2
    # import cv2
    # import time
    # from ultralytics import YOLO

    # model = YOLO('path/to/model')
    # video_path = r'path/to/video'

    # results = model.track(
    #     source=video_path,
    #     # show=True,
    #     half=True,
    #     imgsz=416,
    #     conf=0.75,
    #     vid_stride=1,
    #     # save=True,
    #     verbose=False
    # )

    # max = 0
    # res = None
    # coordinates = (10, 50) # Bottom-left corner
    # font = cv2.FONT_HERSHEY_SIMPLEX
    # fontScale = 1.5
    # outline_color = (255, 255, 255)
    # color = (0, 0, 0) # Green color in BGR
    # thickness = 2

    # mode_changer = StateChanger()
    # for i, r in enumerate(results):
    #     # StateChanger currently adapts only to results provided by YOLO
    #     mode_changer.record_detection(r.boxes.cls.tolist(), r.boxes.xywhn.tolist())
    #     mode_changer.change_state()
    #     cur_state = mode_changer._get_state()

    #     img = r.plot()
    #     cv2.putText(img, cur_state.value['mode'], coordinates, font, fontScale, outline_color, thickness + 4, cv2.LINE_AA)
    #     cv2.putText(img, cur_state.value['mode'], coordinates, font, fontScale, color, thickness, cv2.LINE_AA)
    #     cv2.imshow('bruh', img)
    #     if cv2.waitKey(1) & 0xFF == ord("q"):
    #         break

    #     if len(r.boxes.cls) > max:
    #         res = r
    #         max = len(r.boxes.cls)
    #     time.sleep(0.03)

    # cv2.destroyAllWindows()


# DEBUGGER
    # tmp = StateChanger()
    # tmp.record_detection([6], [[0.5, 0.5, 0.3, 0.9]])
    # tmp.change_state()
    # print(tmp._get_state().value['mode'])